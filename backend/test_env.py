"""Quick smoke test for the updated UAVHybridEnv physics engine."""
from environment import UAVHybridEnv

def test():
    env = UAVHybridEnv(
        engine_size_kw=67.5,         # Reference turboshaft
        battery_capacity_kwh=20.0,   # 20 kWh → 80 kg battery
        target_speed_kmh=250.0,      # 250 km/h cruise
        target_altitude=5000.0,      # 5 km cruise altitude
        payload_weight=200.0,
        use_heuristic_policy=True,
        dt=10.0,                     # Fine 10s steps for accuracy
    )

    print("=== UAV Weight Budget ===")
    print(f"  Airframe:   {env.weight_airframe:.1f} kg")
    print(f"  Payload:    {env.weight_payload:.1f} kg")
    print(f"  Engine:     {env.weight_engine:.1f} kg  ({env.engine_size_kw:.1f} kW)")
    print(f"  Motor:      {env.weight_motor:.1f} kg   (EMRAX 228)")
    print(f"  Battery:    {env.weight_battery:.1f} kg  ({env.battery_capacity_kwh:.1f} kWh)")
    print(f"  Fuel:       {env.fuel_initial:.1f} kg    (remaining budget)")
    print(f"  TOTAL MTOW: {env.mtow:.1f} kg")
    print(f"  Aspect Ratio: {env.aspect_ratio:.2f}")
    print()

    obs, info = env.reset()
    print(f"Initial obs [Alt, Speed, SoC, Fuel, Preq]: {obs}")

    terminated, truncated = False, False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step([0.5])
        steps += 1

    print(f"\n=== Simulation Results ===")
    print(f"  Steps:      {steps}")
    print(f"  Duration:   {env.time_elapsed:.0f}s ({env.time_elapsed/3600:.2f} hrs)")
    print(f"  Final SoC:  {env.soc:.4f}")
    print(f"  Final Fuel: {env.fuel_remaining:.4f} kg")
    print(f"  Reason:     {info.get('reason', 'N/A')}")

    # Phase summary
    phases = {}
    for pt in env.flight_log:
        p = pt["phase"]
        if p not in phases:
            phases[p] = {"start": pt["time"], "end": pt["time"]}
        phases[p]["end"] = pt["time"]
    print(f"\n=== Mission Phase Timeline ===")
    for phase, times in phases.items():
        dur = times["end"] - times["start"]
        print(f"  {phase:12s}: {times['start']/60:6.1f} -> {times['end']/60:6.1f} min  ({dur/60:.1f} min)")

if __name__ == "__main__":
    test()
