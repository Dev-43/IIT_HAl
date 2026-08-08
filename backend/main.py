"""
main.py — FastAPI Backend for Hybrid-Electric UAV Propulsion Optimization Simulator.

Endpoints:
  POST /api/optimize  — Accepts a judge-authored mission profile (ordered legs) plus
                         environmental/design parameters, runs the DEAP GA, returns
                         optimal propulsion sizing + full mission flight telemetry.
  GET  /api/health     — Health check.

Data Flow:
  1. Frontend sends {legs, base_elevation_m, payload_weight, ...}
  2. DEAP GA sizes engine_kw, battery_kwh, and motor_count against that exact mission
     (optionally also the cruise/loiter power-split policy — Outer Loop)
  3. Best individual is re-simulated in UAVHybridEnv with heuristic (or GA-searched)
     power management
  4. Time-series telemetry, optimal specs, and per-leg mission-feasibility results are
     returned as JSON
"""

import os
import json
import math
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from optimizer import optimize_propulsion
from environment import UAVHybridEnv
from rl_policy import extract_rl_features, load_policy, predict_psr, RLPolicyUnavailable

app = FastAPI(
    title="AeroOptima — Hybrid-Electric UAV Propulsion Optimization API",
    description="IIT Indore × HAL Hackathon: 1000 kg Fixed-Wing UAV System Design",
    version="3.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Resolve /data directory path ---- #
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Load motor/aero specs once for response metadata
with open(os.path.join(DATA_DIR, "motor_specs.json"), "r") as f:
    MOTOR_SPECS = json.load(f)
with open(os.path.join(DATA_DIR, "aerodynamics.json"), "r") as f:
    AERO_SPECS = json.load(f)
with open(os.path.join(DATA_DIR, "battery_specs.json"), "r") as f:
    BATTERY_SPECS = json.load(f)


# ---- Pydantic Models ---- #
class MissionLeg(BaseModel):
    role: Literal["cruise", "loiter"]
    altitude_m: float = Field(..., ge=200.0, le=12000.0, description="Absolute altitude, AMSL")
    speed_kmh: float = Field(..., ge=50.0, le=400.0)
    distance_km: Optional[float] = Field(None, gt=0, description="Required for cruise legs")
    duration_min: Optional[float] = Field(None, gt=0, description="Required for loiter legs")
    headwind_kmh: float = Field(0.0, description="Signed: positive=headwind, negative=tailwind for this leg")

    @model_validator(mode="after")
    def validate_role_fields(self):
        if self.role == "cruise" and not self.distance_km:
            raise ValueError("cruise legs require a positive distance_km.")
        if self.role == "loiter" and not self.duration_min:
            raise ValueError("loiter legs require a positive duration_min.")
        if self.headwind_kmh >= self.speed_kmh:
            raise ValueError(
                f"headwind_kmh ({self.headwind_kmh}) must be less than speed_kmh "
                f"({self.speed_kmh}) — groundspeed would be zero or negative."
            )
        return self


class Disturbance(BaseModel):
    """A scripted mid-mission environmental shock: a temperature drop, turbulence spike,
    and/or wind gust active for [trigger_time_min, trigger_time_min + duration_min).
    Purely additive — omitting this field entirely leaves every simulation byte-for-byte
    identical to pre-RL behavior."""
    trigger_time_min: float = Field(..., ge=0.0, description="Minutes into the mission when the shock begins")
    duration_min: float = Field(..., gt=0.0, le=180.0, description="How long the shock lasts, in minutes")
    ambient_temp_c_override: Optional[float] = Field(
        None, ge=-60.0, le=50.0, description="Sea-level temp during the shock; omit to leave temp unchanged"
    )
    turbulence_level_override: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Turbulence level during the shock; omit to leave turbulence unchanged"
    )
    wind_kmh_delta: float = Field(
        0.0, description="Extra headwind (positive) or tailwind (negative) during the shock, added to cruise legs"
    )


class OptimizationRequest(BaseModel):
    legs: list[MissionLeg] = Field(..., min_length=1)
    base_elevation_m: float = Field(0.0, ge=0.0, le=6000.0, description="Home-base elevation, AMSL")
    payload_weight: float = Field(200.0, ge=50.0, le=350.0, description="Payload weight in kg")
    initial_fuel_fraction: float = Field(1.0, ge=0.1, le=1.0)
    ambient_temp_c: float = Field(15.0, ge=-40.0, le=50.0, description="Sea-level ambient temperature")
    turbulence_level: float = Field(0.0, ge=0.0, le=1.0, description="0=calm, 1=rough")
    silent_loiter_mode: bool = Field(True, description="Allow electric-only stealth loiter")
    battery_chemistry: str = Field("Li-NCA", description="Preset key from battery_specs.json chemistry_presets")
    optimize_power_split: bool = Field(
        False, description="Also GA-search cruise/loiter power-split policy (slower, opt-in)"
    )
    policy_mode: Literal["heuristic", "rl"] = Field(
        "heuristic", description="Power-split policy used for the post-GA resimulation/telemetry"
    )
    disturbance: Optional[Disturbance] = Field(
        None, description="Optional scripted mid-mission environmental shock (wind/temp/turbulence)"
    )

    @model_validator(mode="after")
    def validate_leg_altitudes(self):
        for i, leg in enumerate(self.legs):
            if leg.altitude_m <= self.base_elevation_m + 200.0:
                raise ValueError(
                    f"legs[{i}]: altitude_m ({leg.altitude_m}) must be more than 200m above "
                    f"base_elevation_m ({self.base_elevation_m})."
                )
        return self


class LegResult(BaseModel):
    role: str
    required: float
    achieved: float
    completed: bool


class OptimalSpecs(BaseModel):
    engine_kw: float
    battery_kwh: float
    motor_kw: float
    motor_count: int
    endurance_hours: float
    endurance_hours_min: float = None
    endurance_hours_max: float = None
    on_station_min_required: float
    on_station_min_achieved: float
    bonus_reserve_loiter_min: float
    leg_results: list[LegResult]
    reserve_fuel_equivalent_min: float
    reserve_battery_equivalent_min: float
    engine_out_survivable: bool
    empty_weight_kg: float
    empty_weight_fraction: float
    fuel_weight_kg: float
    total_weight_kg: float
    motor_model: str
    engine_weight_kg: float
    motor_weight_kg: float
    battery_weight_kg: float
    battery_chemistry: str
    generator_architecture: str
    generator_efficiency: float
    l_over_d_max: float
    power_split_policy: Optional[dict] = None
    policy_mode: str = "heuristic"


class TelemetryPoint(BaseModel):
    time: float
    altitude: float
    speed: float
    power_required: float
    power_delivered: float
    power_motor: float
    power_engine: float
    soc: float
    fuel: float
    weight: float
    phase: str
    deficit: float
    u: float
    sfc: float = 0.38
    p_aero: float = 0.0
    p_climb: float = 0.0
    climb_rate: float = 0.0
    disturbance_active: bool = False
    leg_index: int = 0


class OptimizationResponse(BaseModel):
    optimal_specs: OptimalSpecs
    telemetry: list[TelemetryPoint]
    generation_stats: list[dict] = []


# ---- Endpoints ---- #
@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_uav(req: OptimizationRequest):
    print(f"\n[API] RECEIVED API CALL: POST /api/optimize")
    print(f"   * legs         : {len(req.legs)}")
    print(f"   * base elev.   : {req.base_elevation_m} m")
    print(f"   * payload      : {req.payload_weight} kg")
    print(f"   * ambient temp : {req.ambient_temp_c}°C, turbulence: {req.turbulence_level}")
    print(f"   * silent loiter: {req.silent_loiter_mode}, chemistry: {req.battery_chemistry}")
    print(f"   * fuel frac    : {req.initial_fuel_fraction * 100:.1f}%")
    print(f"   * optimize power split: {req.optimize_power_split}")
    print(f"   * policy mode  : {req.policy_mode}")
    print(f"   * disturbance  : {req.disturbance.model_dump() if req.disturbance else None}")

    try:
        mission_legs = [leg.model_dump() for leg in req.legs]
        disturbance = req.disturbance.model_dump(exclude_none=True) if req.disturbance else None

        # RL resimulation needs a loaded policy before we spend time on GA sizing —
        # fail fast with a clean 422 rather than doing all that work first.
        rl_model = None
        if req.policy_mode == "rl":
            try:
                rl_model = load_policy()
            except RLPolicyUnavailable as e:
                raise HTTPException(status_code=422, detail=str(e))

        # 1. Run DEAP Genetic Algorithm (Outer Loop) — always sizes with the fast heuristic;
        # policy_mode only changes how the chosen design is resimulated for telemetry below.
        ga_result = optimize_propulsion(
            mission_legs=mission_legs,
            base_elevation_m=req.base_elevation_m,
            payload_weight=req.payload_weight,
            data_dir=DATA_DIR,
            initial_fuel_fraction=req.initial_fuel_fraction,
            ambient_temp_c=req.ambient_temp_c,
            turbulence_level=req.turbulence_level,
            silent_loiter_mode=req.silent_loiter_mode,
            battery_chemistry=req.battery_chemistry,
            optimize_power_split=req.optimize_power_split,
            disturbance=disturbance,
        )

        opt_engine = ga_result["engine_size_kw"]
        opt_battery = ga_result["battery_capacity_kwh"]
        opt_motor_count = ga_result["motor_count"]
        # RL resim always drives PSR via the policy's own action -- a GA-searched
        # phase_psrs (from optimize_power_split) is ignored for this run, since the two
        # are alternative PSR strategies and RL's whole point is a *dynamic* per-step PSR.
        phase_psrs = ga_result.get("phase_psrs") if req.policy_mode == "heuristic" else None

        # 2. Re-simulate best individual with fine time steps for clean telemetry
        print(f"[SIM] SIZING COMPLETE. Re-running dynamic simulation to gather 1-min interval telemetry...")
        print(f"   Using Engine={opt_engine:.2f}kW, Battery={opt_battery:.2f}kWh, Motors={opt_motor_count}")
        print(f"   Policy mode: {req.policy_mode}")

        env = UAVHybridEnv(
            engine_size_kw=opt_engine,
            battery_capacity_kwh=opt_battery,
            motor_count=opt_motor_count,
            mission_legs=mission_legs,
            base_elevation_m=req.base_elevation_m,
            payload_weight=req.payload_weight,
            data_dir=DATA_DIR,
            use_heuristic_policy=(req.policy_mode == "heuristic" and phase_psrs is None),
            phase_psrs=phase_psrs,
            dt=60.0,
            initial_fuel_fraction=req.initial_fuel_fraction,
            ambient_temp_c=req.ambient_temp_c,
            turbulence_level=req.turbulence_level,
            silent_loiter_mode=req.silent_loiter_mode,
            battery_chemistry=req.battery_chemistry,
            disturbance=disturbance,
        )

        obs, info = env.reset()
        terminated, truncated = False, False
        step_count = 0
        while not (terminated or truncated):
            if req.policy_mode == "rl":
                features = extract_rl_features(env, obs)
                action = [predict_psr(rl_model, features)]
            else:
                action = [0.5]  # ignored by the heuristic/phase_psrs policy branches
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        print(f"[SIM] Flight simulation complete:")
        print(f"   * Total steps simulated: {step_count} (dt=60s)")
        print(f"   * Total flight duration: {env.time_elapsed / 3600.0:.2f} hours")
        print(f"   * On-station required/achieved: {env.on_station_min_required:.1f}/{env.on_station_min_achieved:.1f} min")
        print(f"   * Final State of Charge: {env.soc * 100.0:.1f}%")
        print(f"   * Remaining Fuel weight: {env.fuel_remaining:.2f} kg")
        print(f"   * Landing safety status: {info.get('reason', 'Terminated normally')}")

        # 3. Run Sensitivity Sweep
        from sweep import run_sensitivity_sweep
        sweep_res = run_sensitivity_sweep(
            engine_kw=opt_engine,
            battery_kwh=opt_battery,
            motor_count=opt_motor_count,
            mission_legs=mission_legs,
            base_elevation_m=req.base_elevation_m,
            payload_weight=req.payload_weight,
            initial_fuel_fraction=req.initial_fuel_fraction,
            ambient_temp_c=req.ambient_temp_c,
            turbulence_level=req.turbulence_level,
            silent_loiter_mode=req.silent_loiter_mode,
            battery_chemistry=req.battery_chemistry,
            data_dir=DATA_DIR,
            disturbance=disturbance,
        )

        # Analytic max lift-to-drag ratio: L/D_max = 0.5*sqrt(pi*AR*e/CD0)
        cd0 = AERO_SPECS["drag_coefficient_cd0"]
        oswald_e = AERO_SPECS["oswald_efficiency_factor_e"]
        l_over_d_max = 0.5 * math.sqrt(math.pi * env.aspect_ratio * oswald_e / cd0)

        empty_weight_kg = env.weight_empty_and_payload - req.payload_weight
        chemistry_presets = BATTERY_SPECS.get("chemistry_presets", {})
        chemistry_label = chemistry_presets.get(req.battery_chemistry, {}).get("chemistry", req.battery_chemistry)

        specs = OptimalSpecs(
            engine_kw=round(opt_engine, 2),
            battery_kwh=round(opt_battery, 2),
            motor_kw=MOTOR_SPECS["peak_power_kw"] * opt_motor_count,
            motor_count=opt_motor_count,
            endurance_hours=round(env.time_elapsed / 3600.0, 3),
            endurance_hours_min=sweep_res["min_endurance"],
            endurance_hours_max=sweep_res["max_endurance"],
            on_station_min_required=round(env.on_station_min_required, 2),
            on_station_min_achieved=round(env.on_station_min_achieved, 2),
            bonus_reserve_loiter_min=round(env.bonus_reserve_loiter_min, 2),
            leg_results=[LegResult(**r) for r in env.leg_results],
            reserve_fuel_equivalent_min=round(env.reserve_fuel_equivalent_min, 2)
                if env.reserve_fuel_equivalent_min is not None else 0.0,
            reserve_battery_equivalent_min=round(env.reserve_battery_equivalent_min, 2)
                if env.reserve_battery_equivalent_min is not None else 0.0,
            engine_out_survivable=bool(env.engine_out_survivable),
            empty_weight_kg=round(empty_weight_kg, 2),
            empty_weight_fraction=round(empty_weight_kg / env.mtow, 4),
            fuel_weight_kg=round(env.fuel_initial, 2),
            total_weight_kg=round(env.mtow, 2),
            motor_model=f"{MOTOR_SPECS['manufacturer']} {MOTOR_SPECS['model']}",
            engine_weight_kg=round(env.weight_engine, 2),
            motor_weight_kg=round(env.weight_motor, 2),
            battery_weight_kg=round(env.weight_battery, 2),
            battery_chemistry=chemistry_label,
            generator_architecture="800V DC Bus",
            generator_efficiency=BATTERY_SPECS.get("generator_efficiency", 0.85),
            l_over_d_max=round(l_over_d_max, 2),
            power_split_policy=phase_psrs,
            policy_mode=req.policy_mode,
        )

        telemetry = []
        for pt in env.flight_log:
            effective_sfc = env._effective_sfc(pt["power_engine"])
            telemetry.append(TelemetryPoint(sfc=round(effective_sfc, 5), **pt))
        print(f"[API] Response built successfully. Returning {len(telemetry)} telemetry points to frontend UI.\n")

        return OptimizationResponse(
            optimal_specs=specs, telemetry=telemetry, generation_stats=ga_result.get("generation_stats", []),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"[ERROR] DURING API CALL HANDLING: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "motor": MOTOR_SPECS["model"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
