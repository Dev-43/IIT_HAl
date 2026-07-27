import os
import sys
import json
import pytest

# Add backend directory to sys.path to resolve imports correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment import UAVHybridEnv

def load_golden_baseline():
    # NOTE: golden_baseline.json serves as a fixed regression-test fixture to verify the
    # physics simulation core remains consistent. It must NOT be regenerated to track
    # the latest optimum design results.
    baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "golden_baseline.json"))
    if not os.path.exists(baseline_path):
        pytest.fail(f"Golden baseline file not found at {baseline_path}. Run baseline generation first.")
    with open(baseline_path, "r") as f:
        return json.load(f)

GOLDEN_DATA = load_golden_baseline()

@pytest.mark.parametrize("config_key", ["config_1", "config_2", "config_3", "config_4", "config_5"])
def test_regression(config_key):
    config_entry = GOLDEN_DATA[config_key]
    inputs = config_entry["inputs"]
    expected = config_entry["outputs"]

    # Instantiate environment with inputs
    env = UAVHybridEnv(
        engine_size_kw=inputs["engine_kw"],
        battery_capacity_kwh=inputs["battery_kwh"],
        target_speed_kmh=250.0,
        target_altitude=5000.0,
        payload_weight=200.0,
        use_heuristic_policy=True,
        dt=60.0,
        enable_loiter=True,
        initial_fuel_fraction=1.0,
    )

    obs, info = env.reset()
    if env.fuel_initial <= 0.0:
        actual = {
            "endurance_hours": 0.0,
            "total_fuel_burned": 0.0,
            "fuel_trace": [env.fuel_initial],
            "soc_trace": [env.soc],
            "termination_reason": "MTOW exceeded"
        }
    else:
        terminated, truncated = False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step([0.5])
            
        endurance_hours = env.time_elapsed / 3600.0
        fuel_burned = env.fuel_initial - env.fuel_remaining
        soc_trace = [pt["soc"] for pt in env.flight_log]
        fuel_trace = [pt["fuel"] for pt in env.flight_log]
        
        actual = {
            "endurance_hours": endurance_hours,
            "total_fuel_burned": fuel_burned,
            "fuel_trace": fuel_trace,
            "soc_trace": soc_trace,
            "termination_reason": info.get("reason", "unknown")
        }

    # Assertions
    assert actual["termination_reason"] == expected["termination_reason"]
    assert actual["endurance_hours"] == pytest.approx(expected["endurance_hours"], rel=1e-9, abs=1e-9)
    assert actual["total_fuel_burned"] == pytest.approx(expected["total_fuel_burned"], rel=1e-9, abs=1e-9)
    
    assert len(actual["soc_trace"]) == len(expected["soc_trace"])
    for a_soc, e_soc in zip(actual["soc_trace"], expected["soc_trace"]):
        assert a_soc == pytest.approx(e_soc, rel=1e-9, abs=1e-9)
        
    assert len(actual["fuel_trace"]) == len(expected["fuel_trace"])
    for a_fuel, e_fuel in zip(actual["fuel_trace"], expected["fuel_trace"]):
        assert a_fuel == pytest.approx(e_fuel, rel=1e-9, abs=1e-9)
