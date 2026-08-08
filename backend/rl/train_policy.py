"""
train_policy.py — Trains a PPO power-split policy specifically to handle scripted
environmental disturbances (wind gusts, temperature drops, turbulence spikes) that the
static heuristic can't perceive or react to.

Training episodes are short synthetic scenarios (a single cruise or loiter leg, capped at
~3 simulated hours) with a randomized chance of a disturbance at a random point/magnitude/
duration -- full multi-hour GA-sized missions would make CPU training impractical within a
hackathon timeframe, and randomizing the scenario each episode is what makes the policy
generalize instead of memorizing one fixed shock. Calm (no-disturbance) episodes are mixed
in too, so the policy doesn't degrade normal-condition behavior.
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import UAVHybridEnv
from rl_policy import extract_rl_features, normalize_features, RL_FEATURE_DIM, DEFAULT_MODEL_PATH

MAX_EPISODE_STEPS = 200  # at dt=60s, ~3.3 simulated hours -- deliberately short


class DisturbanceTrainingEnv(gym.Env):
    """
    Gymnasium wrapper around UAVHybridEnv for RL training. Each reset() builds a fresh,
    randomized single-leg scenario (and, with probability disturbance_prob, a randomized
    scripted disturbance) so the policy sees a wide distribution of conditions rather than
    one fixed mission. Truncates at MAX_EPISODE_STEPS regardless of the underlying env's
    own termination logic, since a single leg followed by UAVHybridEnv's open-ended
    "extend" reserve loiter would otherwise make episodes far longer than intended.
    """
    metadata = {}

    def __init__(self, engine_kw: float = 90.0, battery_kwh: float = 25.0,
                 disturbance_prob: float = 0.6, dt: float = 60.0, seed: int = None):
        super().__init__()
        self.engine_kw = engine_kw
        self.battery_kwh = battery_kwh
        self.disturbance_prob = disturbance_prob
        self.dt = dt
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(RL_FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._env = None
        self._step_count = 0

    def _random_scenario_legs(self):
        rng = self._rng
        role = "cruise" if rng.random() < 0.5 else "loiter"
        altitude_m = float(rng.uniform(2500, 6000))
        if role == "cruise":
            speed_kmh = float(rng.uniform(150, 280))
            distance_km = float(rng.uniform(100, 300))
            return [{"role": "cruise", "altitude_m": altitude_m, "speed_kmh": speed_kmh, "distance_km": distance_km}]
        speed_kmh = float(rng.uniform(120, 220))
        duration_min = float(rng.uniform(30, 120))
        return [{"role": "loiter", "altitude_m": altitude_m, "speed_kmh": speed_kmh, "duration_min": duration_min}]

    def _random_disturbance(self):
        rng = self._rng
        if rng.random() > self.disturbance_prob:
            return None
        return {
            "trigger_time_min": float(rng.uniform(5, 40)),
            "duration_min": float(rng.uniform(5, 25)),
            "ambient_temp_c_override": float(rng.uniform(-30, -5)),
            "turbulence_level_override": float(rng.uniform(0.5, 1.0)),
            "wind_kmh_delta": float(rng.uniform(20, 60)),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        legs = self._random_scenario_legs()
        disturbance = self._random_disturbance()
        self._env = UAVHybridEnv(
            engine_size_kw=self.engine_kw,
            battery_capacity_kwh=self.battery_kwh,
            mission_legs=legs,
            payload_weight=200.0,
            use_heuristic_policy=False,
            dt=self.dt,
            disturbance=disturbance,
        )
        obs5, info = self._env.reset()
        self._step_count = 0
        features = extract_rl_features(self._env, obs5)
        return normalize_features(features), {}

    def step(self, action):
        psr = float(np.clip(action[0], 0.0, 1.0))
        obs5, reward, terminated, truncated, info = self._env.step([psr])
        self._step_count += 1

        # Extra shaping: prioritize graceful handling specifically during the shock window
        # (the whole point of this policy), on top of the env's own per-step reward.
        shock_penalty = 0.0
        if self._env.disturbance_active and self._env.flight_log:
            shock_penalty = -3.0 * self._env.flight_log[-1]["deficit"]

        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        features = extract_rl_features(self._env, obs5)
        return normalize_features(features), reward + shock_penalty, terminated, truncated, info


def train(total_timesteps: int = 100_000, save_path: str = None, seed: int = 42, verbose: int = 1):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    save_path = save_path or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("========================================================")
    print("[RL] TRAINING POWER-SPLIT POLICY (scripted-disturbance scenarios)")
    print("========================================================")
    print(f"  Algorithm       : PPO (stable-baselines3)")
    print(f"  Feature dim     : {RL_FEATURE_DIM}")
    print(f"  Max episode len : {MAX_EPISODE_STEPS} steps")
    print(f"  Total timesteps : {total_timesteps}")
    print("--------------------------------------------------------")

    env = Monitor(DisturbanceTrainingEnv(seed=seed))
    model = PPO("MlpPolicy", env, verbose=verbose, seed=seed, n_steps=512, batch_size=64)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)

    print(f"\n[SAVE] Trained policy saved to {save_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the RL power-split policy.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, seed=args.seed)
