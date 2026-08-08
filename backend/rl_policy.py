"""
rl_policy.py — RL feature extraction and inference wrapper for the power-split policy.

Deliberately separate from environment.py's Gym contract (observation_space/_get_obs,
used by the GA/heuristic path): this module builds its OWN feature vector from the env's
already-public attributes, so the shipped and tested GA-sizing path is never touched by
anything RL-related.

stable_baselines3/torch are imported lazily, only inside load_policy(), so importing this
module — or anything that imports it, like main.py — never requires the optional `rl`
dependency group unless policy_mode="rl" is actually requested.
"""
import os
import numpy as np

RL_FEATURE_DIM = 9
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "rl_model", "ppo_psr_policy.zip")

# Rough per-feature scale, in the same order as extract_rl_features(), so every input to
# the network is O(1) regardless of physical units -- altitude in meters and SoC as a
# 0-1 fraction otherwise sit on wildly different scales, which hurts training stability.
# Applied identically at training time and inference time (both go through
# normalize_features()) -- extract_rl_features() itself always returns raw physical
# values, so benchmark/debug output stays human-readable.
FEATURE_SCALE = np.array([10000.0, 100.0, 1.0, 1.0, 150.0, 1.0, 50.0, 1.0, 1.0], dtype=np.float32)


def normalize_features(features: np.ndarray) -> np.ndarray:
    return (features / FEATURE_SCALE).astype(np.float32)


class RLPolicyUnavailable(RuntimeError):
    """Raised when policy_mode='rl' is requested but the optional deps or the trained
    model file aren't available. Callers should surface this as a clean error, not a
    stack trace."""
    pass


def extract_rl_features(env, obs: np.ndarray) -> np.ndarray:
    """
    Build the RL policy's 9D feature vector from the env's current public state, given
    the standard 5D Gym observation just returned by env.reset()/env.step(). obs[4] is
    power_required for the step that just completed — reused here rather than
    recomputed, so the two never disagree.

    Order (must match backend/rl/train_policy.py exactly — both sides have to agree):
      [altitude_m, speed_ms, soc, fuel_ratio, power_required_kw,
       sigma (rho/rho0), temp_c, turbulence_level, disturbance_active]
    """
    fuel_ratio = env.fuel_remaining / env.fuel_initial if env.fuel_initial > 0 else 0.0
    rho = env._atmosphere(env.altitude)
    sigma = rho / env.aero["air_density_sea_level_kg_m3"]
    temp_c = env._isa_temperature(env.altitude)
    p_req_kw = float(obs[4])
    return np.array([
        env.altitude,
        env.speed,
        env.soc,
        fuel_ratio,
        p_req_kw,
        sigma,
        temp_c,
        env.turbulence_level,
        1.0 if getattr(env, "disturbance_active", False) else 0.0,
    ], dtype=np.float32)


def load_policy(model_path: str = None):
    """
    Load the trained PPO policy from disk. Raises RLPolicyUnavailable (not an ImportError
    or FileNotFoundError) with a clear, actionable message if the model file or the
    optional dependency group is missing.
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise RLPolicyUnavailable(
            f"No trained RL policy found at {model_path}. Run "
            f"'python backend/rl/train_policy.py' first, or use policy_mode='heuristic'."
        )
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise RLPolicyUnavailable(
            "policy_mode='rl' requires the optional 'rl' dependency group "
            "(stable-baselines3, torch). Install with: uv sync --extra rl"
        ) from e
    return PPO.load(model_path)


def predict_psr(policy, features: np.ndarray) -> float:
    """Run inference and clip to the valid PSR range [0.0, 1.0] (no charging/negative
    mode this round — matches what the heuristic policy actually exercises in practice).
    features should be the RAW vector from extract_rl_features(); normalization is
    applied here so callers never have to remember to do it themselves."""
    action, _ = policy.predict(normalize_features(features), deterministic=True)
    psr = float(np.asarray(action).flatten()[0])
    return float(np.clip(psr, 0.0, 1.0))
