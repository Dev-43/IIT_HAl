# AeroOptima Backend — Hybrid UAV Physics, Optimization & RL Engine

Python simulation, GA sizing, and RL power-split backend for the hybrid-electric UAV.
See the root `README.md` for the project-wide overview and how this fits with the
frontend; this file covers the backend's internals and the full API schema.

---

## 📂 Core Files

* **`environment.py`** — Gymnasium-compliant environment (`UAVHybridEnv`) simulating a
  full multi-leg mission:
  * *Mission sequencer:* executes ordered cruise (by distance) / loiter (by duration)
    legs, auto-RTB on critical fuel/battery, per-leg completion tracking.
  * *Atmosphere:* ISA model for air density ρ(h) and temperature; each accepts an
    optional per-step override so a scripted disturbance can shift them temporarily.
  * *Aerodynamics:* stall speed, Oswald drag polar, Prandtl-Glauert compressibility.
  * *Propulsion:* parallel hybrid power split, motor C-rate limits, turboshaft SFC with
    partial-load penalty, Gagg-Ferrar altitude engine derating, cold-battery derating,
    regenerative descent.
  * *Scripted disturbance:* optional `disturbance` dict — temp/turbulence/wind
    overrides active only inside a configured `[trigger_time_min, trigger_time_min +
    duration_min)` window, otherwise a no-op. Exposed per-step as `disturbance_active`.
* **`optimizer.py`** — DEAP genetic algorithm. 3 base genes (`engine_kw`, `battery_kwh`,
  `motor_count`), optionally +2 more (`psr_cruise`, `psr_loiter`) when
  `optimize_power_split=True`. Fitness = simulated endurance, penalized for an
  incomplete/unsafe mission.
* **`sweep.py`** — runs the same design through ±10% SFC / ±10% battery-density
  scenarios for a sensitivity/robustness range.
* **`main.py`** — FastAPI app: GA sizing → resimulation (heuristic or RL policy) →
  sensitivity sweep → JSON response.
* **`rl_policy.py`** — RL feature extraction (`extract_rl_features`) and PPO inference
  (`load_policy`, `predict_psr`). `stable-baselines3`/`torch` are imported lazily inside
  `load_policy()` only, so importing this module — or anything that imports it, like
  `main.py` — never requires the optional `rl` dependency group unless
  `policy_mode="rl"` is actually requested.
* **`rl/train_policy.py`** — trains the PPO power-split policy on short, randomized
  disturbance scenarios; saves to `rl_model/ppo_psr_policy.zip`.
* **`rl/benchmark_policy.py`** — runs a real, measured heuristic-vs-RL comparison under
  a scripted shock and prints the actual numbers (no hardcoded results).
* **`test_env.py`** / **`test_regression.py`** — pytest suite: physics smoke tests,
  phantom-climb/altitude-derating regressions, and disturbance-window regressions.
* **`ladakh_mission_run.py`** — ad-hoc script validating a realistic high-altitude
  (Nyoma ALG, ~4,130 m AMSL) ISR mission end to end.

---

## 📐 Physics and Governing Equations

### 1. Drag and Lift Coefficients
$$C_L = \frac{2 \cdot m \cdot g \cdot \cos(\gamma)}{\rho \cdot V^2 \cdot S}$$

$$C_D = C_{D0} + \frac{C_L^2}{\pi \cdot AR \cdot e}$$

*Where $C_{D0} = 0.025$, Oswald efficiency $e = 0.85$, and Aspect Ratio $AR = 16.07$ (15m wingspan / 14m² wing area).*

### 2. Shaft Power Required
$$P_{prop} = P_{aero} + P_{climb} = \left(\frac{1}{2}\rho V^3 S C_D\right) + \left(m \cdot g \cdot V_z\right), \qquad P_{shaft} = \frac{P_{prop}}{\eta_{prop}}$$

*Where $\eta_{prop}$ ranges from $0.65$ (takeoff) to $0.85$ (cruise).*

### 3. Engine Altitude Derating (Gagg-Ferrar)
$$\sigma = \rho(h)/\rho_0, \qquad P_{max}(h) = P_{max,SL} \cdot \text{clip}\left(\sigma - \frac{1-\sigma}{7.55},\ 0.15,\ 1.0\right)$$

### 4. Engine SFC Partial-Load Penalty
Fuel consumption scales with throttle load relative to the base SFC ($0.38$ kg/kWh):
* Load fraction ≥ 80%: base SFC
* 50% ≤ Load < 80%: up to +15% worse SFC
* Load < 50%: up to +40% worse SFC

Full formula set (30+ equations, GA hyperparameters, API bounds) is documented in
`D:\IIT_Indore\Docs\AEROTHON_2026_PARAMETERS_AND_FORMULAS.md`.

---

## 📡 API Endpoints

### `POST /api/optimize`
Runs the DEAP GA against the supplied mission, resimulates the winning design at fine
resolution (heuristic or RL policy), runs the sensitivity sweep, and returns everything.

* **Request body:**
  ```json
  {
    "legs": [
      {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300, "headwind_kmh": 0},
      {"role": "loiter", "altitude_m": 3000, "speed_kmh": 180, "duration_min": 60}
    ],
    "base_elevation_m": 0.0,
    "payload_weight": 200.0,
    "initial_fuel_fraction": 1.0,
    "ambient_temp_c": 15.0,
    "turbulence_level": 0.0,
    "silent_loiter_mode": true,
    "battery_chemistry": "Li-NCA",
    "optimize_power_split": false,
    "policy_mode": "heuristic",
    "disturbance": null
  }
  ```
  `legs` is required (min 1); a `cruise` leg needs `distance_km`, a `loiter` leg needs
  `duration_min`. `policy_mode` is `"heuristic"` or `"rl"` — `"rl"` requires a trained
  model at `rl_model/ppo_psr_policy.zip` and the optional `rl` dependency group
  installed (`uv sync --extra rl`), or the API returns a clean 422. `disturbance`, if
  set, is `{trigger_time_min, duration_min, ambient_temp_c_override?,
  turbulence_level_override?, wind_kmh_delta}`.

* **Response body:** `optimal_specs` (engine/battery/motor sizing, endurance +
  sensitivity range, per-leg feasibility results, STANAG-style reserve metrics,
  weight breakdown, echoed `policy_mode`), `telemetry` (per-step time series including
  `disturbance_active`), and `generation_stats` (per-generation GA convergence).

### `GET /api/health`
Basic health check; returns the active motor model.

---

## 🧪 Verification & Testing

```powershell
uv run python -m pytest backend/test_env.py backend/test_regression.py
```

To also verify the RL path end to end (trains a fresh policy — the committed
`rl_model/ppo_psr_policy.zip` already works without this):
```powershell
uv sync --extra rl
uv run backend/rl/train_policy.py --timesteps 50000 --seed 42
uv run backend/rl/benchmark_policy.py
```
