import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import UAVHybridEnv

def run_sensitivity_sweep(
    engine_kw: float,
    battery_kwh: float,
    target_speed_kmh: float = 250.0,
    target_altitude: float = 5000.0,
    payload_weight: float = 200.0,
    enable_loiter: bool = True,
    initial_fuel_fraction: float = 1.0,
    data_dir: str = None,
):
    """
    Run 4 sensitivity simulations for a given design at +/-10% SFC and +/-10% battery density.
    Returns the range of expected endurance.
    """
    scenarios = [
        {"sfc_scale": 1.1, "battery_density_scale": 1.1, "name": "+10% SFC, +10% Battery Density"},
        {"sfc_scale": 1.1, "battery_density_scale": 0.9, "name": "+10% SFC, -10% Battery Density"},
        {"sfc_scale": 0.9, "battery_density_scale": 1.1, "name": "-10% SFC, +10% Battery Density"},
        {"sfc_scale": 0.9, "battery_density_scale": 0.9, "name": "-10% SFC, -10% Battery Density"},
    ]
    
    # Run nominal case
    env_nominal = UAVHybridEnv(
        engine_size_kw=engine_kw,
        battery_capacity_kwh=battery_kwh,
        target_speed_kmh=target_speed_kmh,
        target_altitude=target_altitude,
        payload_weight=payload_weight,
        use_heuristic_policy=True,
        dt=60.0,
        enable_loiter=enable_loiter,
        initial_fuel_fraction=initial_fuel_fraction,
        data_dir=data_dir,
    )
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
        env = UAVHybridEnv(
            engine_size_kw=engine_kw,
            battery_capacity_kwh=battery_kwh,
            target_speed_kmh=target_speed_kmh,
            target_altitude=target_altitude,
            payload_weight=payload_weight,
            use_heuristic_policy=True,
            dt=60.0,
            enable_loiter=enable_loiter,
            initial_fuel_fraction=initial_fuel_fraction,
            data_dir=data_dir,
            sfc_scale=sc["sfc_scale"],
            battery_density_scale=sc["battery_density_scale"],
        )
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

    print(f"Running sensitivity sweep for Engine: {args.engine} kW, Battery: {args.battery} kWh...")
    res = run_sensitivity_sweep(args.engine, args.battery)
    print(f"Nominal: {res['nominal']} hrs")
    print(f"Min: {res['min_endurance']} hrs")
    print(f"Max: {res['max_endurance']} hrs")
