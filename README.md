# AeroOptima — Hybrid-Electric UAV Propulsion Optimization Platform
### IIT Indore × HAL Hackathon (Problem Statement PS-1)

AeroOptima is an AI-driven sizing and simulation platform for the hybrid-electric
propulsion architecture of a **1000 kg tactical fixed-wing UAV**. A genetic algorithm
(outer loop) searches engine size, battery capacity, and motor count; every candidate is
flown through a full multi-leg mission physics simulation (inner loop) before being
scored on endurance. The judge/operator supplies a **mission profile** (ordered
cruise/loiter legs) and environmental conditions — not raw speed/altitude sliders — and
the optimizer decides the propulsion sizing and fuel/battery split needed to fly it.

Power-split decisions during flight are made by either a hand-tuned rule-based heuristic
(default) or a trained reinforcement-learning policy that specifically reacts to sudden
mid-mission environmental shocks (temperature drop, turbulence spike, wind gust) the
heuristic can't perceive — see [RL Power-Split Policy](#-rl-power-split-policy) below.

---

## 🏗️ Project Architecture

```
HALxOPT_D/ (Root)
├── README.md                      # This file
├── pyproject.toml                 # uv-managed deps; optional "rl" extra for RL training/inference
├── data/                           # UAV component specification database
│   ├── aerodynamics.json          # Airframe, drag polar, propeller efficiency
│   ├── battery_specs.json         # Li-ion density/C-rate/SoC limits + chemistry presets (Li-NCA, Li-LFP)
│   ├── motor_specs.json           # EMRAX 228 motor ratings and weight
│   └── turboshaft_specs.json      # Reference turboshaft ratings and SFC curve
├── backend/                        # Python FastAPI + physics simulator + GA + RL
│   ├── environment.py              # Gymnasium flight env (UAVHybridEnv) — mission-leg
│   │                                 sequencer, ISA atmosphere, drag/compressibility,
│   │                                 Gagg-Ferrar engine derating, scripted disturbances
│   ├── optimizer.py                # DEAP genetic algorithm sizing loop
│   ├── sweep.py                    # ±10% SFC/battery-density sensitivity sweep
│   ├── main.py                     # REST API — mission-leg intake, GA sizing, resimulation
│   ├── rl_policy.py                # RL feature extraction + PPO inference wrapper
│   ├── rl/
│   │   ├── train_policy.py         # Trains the PPO power-split policy (stable-baselines3)
│   │   └── benchmark_policy.py     # Real, measured heuristic-vs-RL comparison
│   ├── rl_model/ppo_psr_policy.zip # Trained policy checkpoint (committed)
│   ├── test_env.py, test_regression.py  # pytest suite (physics + disturbance regression)
│   └── ladakh_mission_run.py       # Ad-hoc high-altitude ISR mission validation script
└── frontend/                        # Next.js 16 web dashboard
    └── src/
        ├── components/
        │   ├── Dashboard.tsx        # Mission-leg builder, RL/disturbance controls, state
        │   ├── FlightScene.tsx      # 3D Three.js tactical flight visualizer
        │   ├── TelemetryChart.tsx   # Plotly.js telemetry charts
        │   └── TelemetryTable.tsx   # Scrollable propulsion status matrix
        └── app/page.tsx             # Entrypoint
```

---

## ⚡ Running Locally

Open two terminal sessions from the project root.

### 1. Backend (FastAPI)
```powershell
# Terminal 1
uv run python backend/main.py
```
Starts at **`http://127.0.0.1:8000`** (auto-reloads on file changes). To also enable
`policy_mode="rl"` (optional — the heuristic default works without this):
```powershell
uv sync --extra rl   # installs torch + stable-baselines3, one-time
```

### 2. Frontend (Next.js)
```powershell
# Terminal 2
cd frontend
npm install
npm run dev
```
Starts at **`http://localhost:3000`**. Two "Quick Load" demo presets are available in
the Simulation Constraints panel ("Nyoma ISR Patrol", "Storm-Front Contingency") that
pre-fill a full mission + policy/disturbance configuration for fast live demos.

---

## ⚙️ Sizing & Optimization Framework

1. **Outer loop (DEAP genetic algorithm)** — searches, per the actual mission supplied:
   * **Engine shaft rating:** 30–120 kW
   * **Battery pack capacity:** 16.69–50 kWh (floor set by the STANAG 4671 30-min
     battery-only reserve calculation, not an arbitrary minimum)
   * **Motor count:** {1, 2, 4} EMRAX 228 units
   * *(optional, `--optimize-power-split` / `optimize_power_split: true`)* two more
     genes for a GA-searched cruise/loiter power-split ratio, instead of the heuristic
   Population 40, generations 15, `cxBlend` crossover, `mutGaussian` mutation,
   tournament selection.
2. **Inner loop (Gymnasium environment, `UAVHybridEnv`)** — flies every GA candidate
   through the *exact* mission-leg sequence supplied (see below) under ISA atmosphere,
   Oswald drag + Prandtl-Glauert compressibility, Gagg-Ferrar altitude engine derating,
   SFC partial-load penalty, cold-battery derating, and regenerative descent.
3. **Fitness** = total endurance (hours); ×0.4 penalty if the aircraft doesn't land
   safely, or any loiter leg's on-station time requirement isn't fully met; an
   MTOW-exceeding individual scores 0 immediately.

---

## 🗺️ Mission-Leg Engine

The optimizer targets a **judge-authored mission**, not generic sliders:
- **Cruise legs**: altitude, speed, and a required distance (km) to cover, optionally
  with headwind/tailwind.
- **Loiter legs**: altitude, speed, and a required on-station duration (min) — can run
  in "silent loiter" mode (electric-only, engine off) for stealth.
- Legs execute in order; a leg only counts as complete once its distance/duration
  requirement is fully met; the aircraft auto-RTBs if fuel/battery become critical
  before that.
- `base_elevation_m` sets the home-base altitude (e.g. a high-altitude airstrip); every
  leg altitude must clear a 200 m obstacle-clearance margin above it.
- `battery_chemistry` (`Li-NCA` / `Li-LFP`) is a request-level preset, not GA-searched —
  it's a categorical engineering choice, not a continuous dial.

---

## 🤖 RL Power-Split Policy

The heuristic power-split policy reacts to phase/SoC/fuel-ratio — it has no way to
perceive a *sudden* environmental change mid-flight. The RL policy exists specifically
to handle that case:

- **Scripted disturbance**: an optional `disturbance` (trigger time, duration, temp
  drop, turbulence spike, wind gust) can be attached to a request — active only inside
  its time window, zero effect otherwise. Purely additive; omitting it leaves every
  simulation byte-identical to the no-RL path.
- **`policy_mode: "heuristic" | "rl"`** on `POST /api/optimize` — GA sizing *always*
  uses the fast heuristic regardless of this flag; `policy_mode` only changes how the
  chosen design is resimulated for telemetry/comparison afterward.
- Train: `uv run backend/rl/train_policy.py --timesteps 100000` (PPO via
  `stable-baselines3`, on short randomized-disturbance scenarios; saves to
  `backend/rl_model/ppo_psr_policy.zip`, which is committed so the API works without
  requiring a fresh train).
- Benchmark: `uv run backend/rl/benchmark_policy.py` — reports a real, measured
  heuristic-vs-RL comparison under a scripted shock. Last measured result: RL cuts
  battery SoC drop during the shock from 0.0565 to 0.0039 vs. the heuristic, at the cost
  of slightly more fuel burned (5.93 kg → 6.17 kg) and marginally lower total endurance
  — a genuine trade-off (battery preservation vs. fuel), not an unqualified win.

---

## 🔒 Military Airworthiness Compliance (STANAG 4671)

- **Battery floor**: `battery_specs.json`'s 16.69 kWh minimum was derived by hand from
  an assumed average loiter draw (27.24 kW / 96% motor efficiency / 85% usable SoC
  window) to cover a 30-minute battery-only reserve — the GA independently converges to
  exactly this floor when it's the binding constraint.
- **Live reserve metrics**: every run, the environment computes
  `reserve_fuel_equivalent_min`, `reserve_battery_equivalent_min`, and
  `engine_out_survivable` (whether the reserve at the flight's *minimum* SoC still
  clears ≥30 min) directly from the simulated flight, not from the hand-calc above.
- **Be aware these two don't always agree.** Example — the default sample mission
  (`cruise 300km → loiter 60min → cruise 300km`), sized with a fixed seed
  (`uv run backend/optimizer.py --seed 7`): engine 115.0 kW, battery 16.69 kWh, motor
  count 1, endurance 20.43 h — but the *live* `reserve_battery_equivalent_min` for this
  specific mission is only ~20.5 min, so `engine_out_survivable` is **False** for this
  run. The battery floor guarantees the GA won't search below 16.69 kWh; it does not by
  itself guarantee every mission profile clears the 30-minute live check — that depends
  on the specific mission's SoC trajectory and must be read per-run, not assumed.
- **Input validation**: Pydantic rejects out-of-envelope requests (altitude/speed/temp
  bounds, leg role field requirements) before they reach the simulator.

---

## 📡 API (brief — see `backend/README.md` for the full request/response schema)

- **`POST /api/optimize`** — body: `{legs: [...], base_elevation_m, payload_weight,
  ambient_temp_c, turbulence_level, silent_loiter_mode, battery_chemistry,
  optimize_power_split, policy_mode, disturbance}`. Returns optimal sizing specs, a
  sensitivity range, per-leg mission-feasibility results, and full flight telemetry.
- **`GET /api/health`** — basic health check.

---

## 🧪 Reproducing Results

### 1. Baseline 3-gene GA sizing (engine, battery, motor count)
```powershell
uv run backend/optimizer.py --seed 7
```

### 2. 5-gene GA (hardware + power-split ratio co-optimized)
```powershell
uv run backend/optimizer.py --optimize-power-split --seed 7
```

### 3. Sensitivity sweep on a given design
```powershell
uv run backend/sweep.py --engine 115.0 --battery 16.69
```

### 4. Train + benchmark the RL power-split policy
```powershell
uv run backend/rl/train_policy.py --timesteps 100000 --seed 42
uv run backend/rl/benchmark_policy.py
```

### 5. Full regression + physics test suite
```powershell
uv run python -m pytest backend/test_env.py backend/test_regression.py
```
