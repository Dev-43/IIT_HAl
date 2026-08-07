"""Quick smoke test for the updated UAVHybridEnv physics engine."""
from environment import UAVHybridEnv

def test():
    env = UAVHybridEnv(
        engine_size_kw=120.0,        # Increased size to handle altitude derating
        battery_capacity_kwh=40.0,   # Increased capacity
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
    print()

    # Step 1.4: Engine altitude derating checks
    print("=== Step 1.4: Engine Altitude Derating Checks ===")
    p_sl = env._max_engine_power(0.0, is_peak=False)
    p_sl_peak = env._max_engine_power(0.0, is_peak=True)
    print(f"  Sea-level continuous: {p_sl:.3f} kW (Expected: {env.engine_continuous_kw:.3f} kW)")
    print(f"  Sea-level peak:       {p_sl_peak:.3f} kW (Expected: {env.engine_peak_kw:.3f} kW)")
    assert abs(p_sl - env.engine_continuous_kw) < 1e-6, "Sea-level power mismatch"
    assert abs(p_sl_peak - env.engine_peak_kw) < 1e-6, "Sea-level peak power mismatch"
    
    p_3000 = env._max_engine_power(3000.0, is_peak=False)
    print(f"  3000m continuous:     {p_3000:.3f} kW")
    assert p_3000 < p_sl, "Altitude power lapse failed to decrease power at 3000m"
    
    p_ceiling = env._max_engine_power(env.target_altitude, is_peak=False)
    print(f"  Service Ceiling ({env.target_altitude}m): {p_ceiling:.3f} kW")
    assert p_ceiling > 0.0, "Engine power at service ceiling is non-positive"
    print("  Step 1.4 checks PASSED!")
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

def test_step2_phantom_climb_fixes():
    print("=== Step 2.5: Phantom Climb Bug Validation ===")
    
    # Case A: Sufficiently sized configuration (P_excess > 0 throughout climb)
    print("  Running Case A: Adequate power sizing...")
    env_good = UAVHybridEnv(
        engine_size_kw=80.0,
        battery_capacity_kwh=30.0,
        target_speed_kmh=250.0,
        target_altitude=5000.0,
        payload_weight=200.0,
        use_heuristic_policy=True,
        dt=10.0,
    )
    obs, info = env_good.reset()
    terminated, truncated = False, False
    climb_phases = [env_good.PHASE_TAKEOFF, env_good.PHASE_CLIMB]
    
    climb_positions_ok = True
    while not (terminated or truncated) and env_good.current_phase in climb_phases:
        prev_alt = env_good.altitude
        obs, reward, terminated, truncated, info = env_good.step([0.5])
        if env_good.current_phase in climb_phases:
            # Altitude must increase
            if env_good.altitude <= prev_alt:
                climb_positions_ok = False
                print(f"    ERROR: Altitude did not increase in {env_good.current_phase}! Alt: {prev_alt} -> {env_good.altitude}")
                
    assert climb_positions_ok, "Case A failed: Altitude failed to increase during climb phase under adequate power"
    print("    Case A Passed: Aircraft climbed successfully.")

    # Case B: Severely undersized power (P_excess < 0 at some point)
    print("  Running Case B: Undersized power...")
    env_under = UAVHybridEnv(
        engine_size_kw=20.0, # extremely undersized
        battery_capacity_kwh=2.0, # extremely undersized
        target_speed_kmh=250.0,
        target_altitude=5000.0,
        payload_weight=200.0,
        use_heuristic_policy=True,
        dt=10.0,
    )
    obs, info = env_under.reset()
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        prev_alt = env_under.altitude
        obs, reward, terminated, truncated, info = env_under.step([0.5])
        # If climb was prevented, check that altitude did not increase
        if env_under.climb_prevented:
            if env_under.altitude > prev_alt:
                assert False, f"ERROR: Aircraft climbed when climb was prevented! Alt: {prev_alt} -> {env_under.altitude}"

    print("    Case B Passed: No climb allowed under power deficit, and simulation completed safely.")
    print("  Step 2.5 checks PASSED!")
    print()

if __name__ == "__main__":
    test()
    test_step2_phantom_climb_fixes()
