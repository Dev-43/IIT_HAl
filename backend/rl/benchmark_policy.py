"""
benchmark_policy.py — Honest heuristic-vs-RL comparison under a scripted mid-mission
disturbance. Reports real measured numbers only -- no hardcoded/assumed results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import UAVHybridEnv
from rl_policy import extract_rl_features, load_policy, predict_psr


def run_mission(mission_legs, disturbance, engine_kw, battery_kwh, policy_mode="heuristic",
                 model=None, dt=60.0):
    env = UAVHybridEnv(
        engine_size_kw=engine_kw,
        battery_capacity_kwh=battery_kwh,
        mission_legs=mission_legs,
        payload_weight=200.0,
        use_heuristic_policy=(policy_mode == "heuristic"),
        dt=dt,
        disturbance=disturbance,
    )
    obs, info = env.reset()
    terminated, truncated = False, False
    max_deficit_during = 0.0
    deficit_steps_during = 0
    climb_prevented_steps_during = 0
    soc_at_disturbance_start = None
    fuel_at_disturbance_start = None
    soc_drop_during = 0.0
    fuel_burn_during = 0.0

    while not (terminated or truncated):
        if policy_mode == "rl":
            features = extract_rl_features(env, obs)
            action = [predict_psr(model, features)]
        else:
            action = [0.5]  # ignored by the heuristic policy
        obs, reward, terminated, truncated, info = env.step(action)
        if env.disturbance_active:
            if soc_at_disturbance_start is None:
                soc_at_disturbance_start = env.soc
                fuel_at_disturbance_start = env.fuel_remaining
            if env.climb_prevented:
                climb_prevented_steps_during += 1
            if env.flight_log:
                deficit = env.flight_log[-1]["deficit"]
                max_deficit_during = max(max_deficit_during, deficit)
                if deficit > 0.5:
                    deficit_steps_during += 1
        elif soc_at_disturbance_start is not None and soc_drop_during == 0.0:
            # first step after the window closes -- capture the net change across it
            soc_drop_during = soc_at_disturbance_start - env.soc
            fuel_burn_during = fuel_at_disturbance_start - env.fuel_remaining

    reason = info.get("reason", "unknown")
    return {
        "reason": reason,
        "landed": "Landed" in reason or "Mission completed" in reason,
        "endurance_hours": round(env.time_elapsed / 3600.0, 3),
        "final_soc": round(env.soc, 4),
        "final_fuel_kg": round(env.fuel_remaining, 2),
        "on_station_required_min": round(env.on_station_min_required, 1),
        "on_station_achieved_min": round(env.on_station_min_achieved, 1),
        "max_deficit_kw_during_disturbance": round(max_deficit_during, 3),
        "deficit_steps_during_disturbance": deficit_steps_during,
        "climb_prevented_steps_during_disturbance": climb_prevented_steps_during,
        "soc_drop_during_disturbance": round(soc_drop_during, 4),
        "fuel_burn_during_disturbance_kg": round(fuel_burn_during, 3),
    }


def benchmark(engine_kw: float = 118.0, battery_kwh: float = 22.0, model_path: str = None):
    mission_legs = [
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300},
        {"role": "loiter", "altitude_m": 3000, "speed_kmh": 180, "duration_min": 120},
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300},
    ]
    # Triggered during climb (0-~15min) deliberately: it's the phase with the least power
    # margin (Gagg-Ferrar derating + the climb battery-preservation guard capping PSR at
    # 0.05), so a shock there is the most likely to produce a measurable difference. A
    # scripted disturbance in level cruise/loiter tends to get absorbed without a visible
    # effect unless the aircraft is undersized to the point of failing regardless of any
    # disturbance -- this default was chosen empirically for that reason.
    disturbance = {
        "trigger_time_min": 2.0,
        "duration_min": 12.0,
        "ambient_temp_c_override": -35.0,
        "turbulence_level_override": 1.0,
        "wind_kmh_delta": 90.0,
    }

    print("=" * 60)
    print(" AeroOptima: Heuristic vs RL — Scripted-Disturbance Benchmark")
    print(f" Engine: {engine_kw}kW, Battery: {battery_kwh}kWh")
    print("=" * 60)

    heuristic_result = run_mission(mission_legs, disturbance, engine_kw, battery_kwh, policy_mode="heuristic")
    print("\n[Heuristic policy]")
    for k, v in heuristic_result.items():
        print(f"  {k}: {v}")

    try:
        model = load_policy(model_path)
    except Exception as e:
        print(f"\n[RL] Could not load trained policy: {e}")
        print("Run train_policy.py first to produce a comparison.")
        return {"heuristic": heuristic_result, "rl": None}

    rl_result = run_mission(mission_legs, disturbance, engine_kw, battery_kwh, policy_mode="rl", model=model)
    print("\n[RL policy]")
    for k, v in rl_result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print(" COMPARISON (measured, not assumed)")
    print("=" * 60)
    print(f"  Endurance                : heuristic={heuristic_result['endurance_hours']}h  vs  rl={rl_result['endurance_hours']}h")
    print(f"  Landed successfully      : heuristic={heuristic_result['landed']}  vs  rl={rl_result['landed']}")
    print(f"  Max deficit in shock     : heuristic={heuristic_result['max_deficit_kw_during_disturbance']}kW  vs  rl={rl_result['max_deficit_kw_during_disturbance']}kW")
    print(f"  Deficit steps in shock   : heuristic={heuristic_result['deficit_steps_during_disturbance']}  vs  rl={rl_result['deficit_steps_during_disturbance']}")
    print(f"  Climb-prevented in shock : heuristic={heuristic_result['climb_prevented_steps_during_disturbance']}  vs  rl={rl_result['climb_prevented_steps_during_disturbance']}")
    print(f"  SoC drop across shock    : heuristic={heuristic_result['soc_drop_during_disturbance']}  vs  rl={rl_result['soc_drop_during_disturbance']}")
    print(f"  Fuel burned across shock : heuristic={heuristic_result['fuel_burn_during_disturbance_kg']}kg  vs  rl={rl_result['fuel_burn_during_disturbance_kg']}kg")
    print("=" * 60)

    return {"heuristic": heuristic_result, "rl": rl_result}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark RL vs heuristic under a scripted disturbance.")
    parser.add_argument("--engine", type=float, default=118.0)
    parser.add_argument("--battery", type=float, default=22.0)
    args = parser.parse_args()
    benchmark(engine_kw=args.engine, battery_kwh=args.battery)
