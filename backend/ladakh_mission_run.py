"""
Ad-hoc script: GA-size the UAV against a realistic Indian-military high-altitude
LAC (Ladakh) persistent-ISR mission profile, then re-simulate the winner at 60s
telemetry resolution and print the mission-level summary (leg completion, reserve
minutes, engine-out survivability).

Base: Nyoma Advanced Landing Ground, Ladakh (~4,130 m AMSL, one of the world's
highest operational airstrips, in active IAF/Army use).
"""
import random
import numpy as np
from optimizer import optimize_propulsion
from environment import UAVHybridEnv

random.seed(42)
np.random.seed(42)

BASE_ELEVATION_M = 4130.0  # Nyoma ALG, Ladakh

NYOMA_LAC_ISR_MISSION = [
    {"role": "cruise", "altitude_m": 6500, "speed_kmh": 250, "distance_km": 80, "headwind_kmh": 15.0},
    {"role": "loiter", "altitude_m": 5500, "speed_kmh": 170, "duration_min": 720.0},  # 12h guaranteed on-station ISR
    {"role": "cruise", "altitude_m": 6500, "speed_kmh": 250, "distance_km": 80, "headwind_kmh": 0.0},
]

COMMON = dict(
    mission_legs=NYOMA_LAC_ISR_MISSION,
    base_elevation_m=BASE_ELEVATION_M,
    payload_weight=200.0,
    initial_fuel_fraction=1.0,
    ambient_temp_c=-5.0,       # cold-soak winter LAC baseline (sea-level-equivalent, feeds lapse rate)
    silent_loiter_mode=False,  # 12h watch must run on the hybrid heuristic split, not full-battery stealth
)

print("=" * 70)
print("GA sizing against Nyoma -> LAC patrol box -> Nyoma ISR mission")
print("=" * 70)
result = optimize_propulsion(**COMMON, pop_size=40, n_gen=15)
print("\nOptimization complete!")
print(f"  Engine:      {result['engine_size_kw']:.2f} kW")
print(f"  Battery:     {result['battery_capacity_kwh']:.2f} kWh")
print(f"  Motor Count: {result['motor_count']}")
print(f"  Endurance:   {result['fitness']:.4f} hours")

# Re-simulate the GA winner at 60s telemetry resolution for the mission summary
env = UAVHybridEnv(
    engine_size_kw=result["engine_size_kw"],
    battery_capacity_kwh=result["battery_capacity_kwh"],
    motor_count=result["motor_count"],
    dt=60.0,
    use_heuristic_policy=True,
    **COMMON,
)
obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step([0.5])

print("\n" + "=" * 70)
print("MISSION SUMMARY")
print("=" * 70)
print(f"  Termination reason:          {info.get('reason')}")
print(f"  Total mission time:          {env.time_elapsed/3600.0:.2f} h")
print(f"  On-station required (min):   {env.on_station_min_required:.1f}")
print(f"  On-station achieved (min):   {env.on_station_min_achieved:.1f}")
print(f"  Bonus reserve loiter (min):  {env.bonus_reserve_loiter_min:.1f}")
print(f"  Fuel remaining (kg):         {env.fuel_remaining:.2f}")
print(f"  Final battery SoC:           {env.soc:.3f}")
print(f"  Min SoC seen in flight:      {env._min_soc_seen:.3f}")
print(f"  Reserve fuel equivalent min: {env.reserve_fuel_equivalent_min:.1f}" if env.reserve_fuel_equivalent_min is not None else "  Reserve fuel equivalent min: n/a")
print(f"  Reserve battery equiv. min:  {env.reserve_battery_equivalent_min:.1f}" if env.reserve_battery_equivalent_min is not None else "  Reserve battery equiv. min:  n/a")
print(f"  Engine-out survivable(>=30min): {env.engine_out_survivable}")
print(f"  Weight budget: airframe={env.weight_airframe:.1f} payload={env.weight_payload:.1f} "
      f"engine={env.weight_engine:.1f} motor={env.weight_motor:.1f} battery={env.weight_battery:.1f} "
      f"fuel_initial={env.fuel_initial:.1f}")
print(f"  Leg results: {env.leg_results}")
