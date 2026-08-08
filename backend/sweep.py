import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import UAVHybridEnv


def run_sensitivity_sweep(
    engine_kw: float,
    battery_kwh: float,
    mission_legs: list,
    motor_count: int = 1,
    base_elevation_m: float = 0.0,
    payload_weight: float = 200.0,
    initial_fuel_fraction: float = 1.0,
    ambient_temp_c: float = 15.0,
    turbulence_level: float = 0.0,
    silent_loiter_mode: bool = True,
    battery_chemistry: str = "Li-NCA",
    data_dir: str = None,
    disturbance: dict = None,
):
    """
    Run 4 sensitivity simulations for a given design at +/-10% SFC and +/-10% battery
    density, against the same mission profile used to size the design. Returns the range
    of expected endurance. Always uses the heuristic policy (this is a sizing-robustness
    check, not a policy comparison) but still honors `disturbance` so the sweep reflects
    the same scripted shock, if any, that the headline resimulation used.
    """
    scenarios = [
        {"sfc_scale": 1.1, "battery_density_scale": 1.1, "name": "+10% SFC, +10% Battery Density"},
        {"sfc_scale": 1.1, "battery_density_scale": 0.9, "name": "+10% SFC, -10% Battery Density"},
        {"sfc_scale": 0.9, "battery_density_scale": 1.1, "name": "-10% SFC, +10% Battery Density"},
        {"sfc_scale": 0.9, "battery_density_scale": 0.9, "name": "-10% SFC, -10% Battery Density"},
    ]

    def _build_env(**overrides):
        return UAVHybridEnv(
            engine_size_kw=engine_kw,
            battery_capacity_kwh=battery_kwh,
            motor_count=motor_count,
            mission_legs=mission_legs,
            base_elevation_m=base_elevation_m,
            payload_weight=payload_weight,
            use_heuristic_policy=True,
            dt=60.0,
            initial_fuel_fraction=initial_fuel_fraction,
            ambient_temp_c=ambient_temp_c,
            turbulence_level=turbulence_level,
            silent_loiter_mode=silent_loiter_mode,
            battery_chemistry=battery_chemistry,
            data_dir=data_dir,
            disturbance=disturbance,
            **overrides,
        )

    # Run nominal case
    env_nominal = _build_env()
    env_nominal.reset()
    if env_nominal.fuel_initial <= 0.0:
        nominal_endurance = 0.0
    else:
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env_nominal.step([0.5])
        nominal_endurance = env_nominal.time_elapsed / 3600.0

    results = []
    endurances = []

    for sc in scenarios:
        env = _build_env(sfc_scale=sc["sfc_scale"], battery_density_scale=sc["battery_density_scale"])
        env.reset()
        if env.fuel_initial <= 0.0:
            endurance = 0.0
        else:
            terminated, truncated = False, False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step([0.5])
            endurance = env.time_elapsed / 3600.0

        results.append({
            "name": sc["name"],
            "sfc_scale": sc["sfc_scale"],
            "battery_density_scale": sc["battery_density_scale"],
            "endurance_hours": round(endurance, 3),
        })
        endurances.append(endurance)

    return {
        "nominal": round(nominal_endurance, 3),
        "cases": results,
        "min_endurance": round(min(endurances), 3),
        "max_endurance": round(max(endurances), 3),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run sensitivity sweep for UAV sizing.")
    parser.add_argument("--engine", type=float, required=True, help="Engine power size in kW.")
    parser.add_argument("--battery", type=float, required=True, help="Battery capacity in kWh.")
    args = parser.parse_args()

    SAMPLE_MISSION = [
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300},
        {"role": "loiter", "altitude_m": 3000, "speed_kmh": 180, "duration_min": 60},
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300},
    ]

    print(f"Running sensitivity sweep for Engine: {args.engine} kW, Battery: {args.battery} kWh...")
    res = run_sensitivity_sweep(args.engine, args.battery, mission_legs=SAMPLE_MISSION)
    print(f"Nominal: {res['nominal']} hrs")
    print(f"Min: {res['min_endurance']} hrs")
    print(f"Max: {res['max_endurance']} hrs")
