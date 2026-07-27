"""
main.py — FastAPI Backend for Hybrid-Electric UAV Propulsion Optimization Simulator.

Endpoints:
  POST /api/optimize  — Accepts mission parameters, runs DEAP GA, returns optimal
                         propulsion sizing + full mission flight telemetry.
  GET  /api/health     — Health check.

Data Flow:
  1. Frontend sends {target_speed_kmh, target_altitude, payload_weight}
  2. DEAP GA sizes engine_kw and battery_kwh (Outer Loop)
  3. Best individual is re-simulated in UAVHybridEnv with heuristic power management
  4. Time-series telemetry and optimal specs returned as JSON
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from optimizer import optimize_propulsion
from environment import UAVHybridEnv

app = FastAPI(
    title="AeroOptima — Hybrid-Electric UAV Propulsion Optimization API",
    description="IIT Indore × HAL Hackathon: 1000 kg Fixed-Wing UAV System Design",
    version="2.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Resolve /data directory path ---- #
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Load motor specs once for response metadata
with open(os.path.join(DATA_DIR, "motor_specs.json"), "r") as f:
    MOTOR_SPECS = json.load(f)


# ---- Pydantic Models ---- #
class OptimizationRequest(BaseModel):
    target_speed_kmh: float = Field(
        250.0,
        description="Target cruise speed in km/h",
        ge=100.0,
        le=400.0,
    )
    target_altitude: float = Field(
        5000.0,
        description="Target cruise altitude in meters",
        ge=500.0,
        le=10000.0,
    )
    payload_weight: float = Field(
        200.0,
        description="Payload weight in kg",
        ge=50.0,
        le=350.0,
    )
    enable_loiter: bool = Field(
        True,
        description="Toggle loiter holding phase vs straight flight"
    )
    initial_fuel_fraction: float = Field(
        1.0,
        description="Fraction of max fuel capacity to start with (0.1–1.0)",
        ge=0.1,
        le=1.0,
    )


class OptimalSpecs(BaseModel):
    engine_kw: float
    battery_kwh: float
    motor_kw: float
    endurance_hours: float
    endurance_hours_min: float = None
    endurance_hours_max: float = None
    empty_weight_kg: float
    fuel_weight_kg: float
    total_weight_kg: float
    motor_model: str
    engine_weight_kg: float
    motor_weight_kg: float
    battery_weight_kg: float


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


class OptimizationResponse(BaseModel):
    optimal_specs: OptimalSpecs
    telemetry: list[TelemetryPoint]


# ---- Endpoints ---- #
@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_uav(req: OptimizationRequest):
    print(f"\n[API] RECEIVED API CALL: POST /api/optimize")
    print(f"   * speed   : {req.target_speed_kmh} km/h")
    print(f"   * altitude: {req.target_altitude} m")
    print(f"   * payload : {req.payload_weight} kg")
    print(f"   * loiter  : {req.enable_loiter}")
    print(f"   * fuel frac: {req.initial_fuel_fraction * 100:.1f}%")
    
    try:
        # 1. Run DEAP Genetic Algorithm (Outer Loop)
        ga_result = optimize_propulsion(
            target_speed_kmh=req.target_speed_kmh,
            target_altitude=req.target_altitude,
            payload_weight=req.payload_weight,
            data_dir=DATA_DIR,
            enable_loiter=req.enable_loiter,
            initial_fuel_fraction=req.initial_fuel_fraction,
        )

        opt_engine = ga_result["engine_size_kw"]
        opt_battery = ga_result["battery_capacity_kwh"]

        # 2. Re-simulate best individual with fine time steps for clean telemetry
        print(f"[SIM] SIZING COMPLETE. Re-running dynamic simulation to gather 1-min interval telemetry...")
        print(f"   Using Engine = {opt_engine:.2f} kW, Battery = {opt_battery:.2f} kWh")
        
        env = UAVHybridEnv(
            engine_size_kw=opt_engine,
            battery_capacity_kwh=opt_battery,
            target_speed_kmh=req.target_speed_kmh,
            target_altitude=req.target_altitude,
            payload_weight=req.payload_weight,
            data_dir=DATA_DIR,
            use_heuristic_policy=True,
            dt=60.0,
            enable_loiter=req.enable_loiter,
            initial_fuel_fraction=req.initial_fuel_fraction,
        )

        obs, info = env.reset()
        terminated, truncated = False, False
        step_count = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step([0.5])
            step_count += 1

        print(f"[SIM] Flight simulation complete:")
        print(f"   * Total steps simulated: {step_count} (dt=60s)")
        print(f"   * Total flight duration: {env.time_elapsed / 3600.0:.2f} hours")
        print(f"   * Final State of Charge: {env.soc * 100.0:.1f}%")
        print(f"   * Remaining Fuel weight: {env.fuel_remaining:.2f} kg")
        print(f"   * Landing safety status: {info.get('reason', 'Terminated normally')}")

        # 3. Run Sensitivity Sweep
        from sweep import run_sensitivity_sweep
        sweep_res = run_sensitivity_sweep(
            engine_kw=opt_engine,
            battery_kwh=opt_battery,
            target_speed_kmh=req.target_speed_kmh,
            target_altitude=req.target_altitude,
            payload_weight=req.payload_weight,
            enable_loiter=req.enable_loiter,
            initial_fuel_fraction=req.initial_fuel_fraction,
            data_dir=DATA_DIR
        )

        specs = OptimalSpecs(
            engine_kw=round(opt_engine, 2),
            battery_kwh=round(opt_battery, 2),
            motor_kw=MOTOR_SPECS["peak_power_kw"],
            endurance_hours=round(env.time_elapsed / 3600.0, 3),
            endurance_hours_min=sweep_res["min_endurance"],
            endurance_hours_max=sweep_res["max_endurance"],
            empty_weight_kg=round(env.weight_empty_and_payload - req.payload_weight, 2),
            fuel_weight_kg=round(env.fuel_initial, 2),
            total_weight_kg=round(env.mtow, 2),
            motor_model=f"{MOTOR_SPECS['manufacturer']} {MOTOR_SPECS['model']}",
            engine_weight_kg=round(env.weight_engine, 2),
            motor_weight_kg=round(env.weight_motor, 2),
            battery_weight_kg=round(env.weight_battery, 2),
        )

        telemetry = []
        for pt in env.flight_log:
            effective_sfc = env._effective_sfc(pt["power_engine"])
            telemetry.append(TelemetryPoint(sfc=round(effective_sfc, 5), **pt))
        print(f"[API] Response built successfully. Returning {len(telemetry)} telemetry points to frontend UI.\n")

        return OptimizationResponse(optimal_specs=specs, telemetry=telemetry)

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
