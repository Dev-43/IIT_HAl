# AeroOptima — Full Codebase Report
> **Target audience:** Any code agent (Claude Code, Codex, Gemini, etc.) that needs a complete, parameter-level understanding of this repository before making changes.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Layer — `/data/`](#3-data-layer--data)
4. [Backend — `/backend/`](#4-backend--backend)
   - 4.1 [environment.py — Gymnasium Simulation](#41-environmentpy--gymnasium-simulation)
   - 4.2 [optimizer.py — DEAP Genetic Algorithm](#42-optimizerpy--deap-genetic-algorithm)
   - 4.3 [sweep.py — Sensitivity Analysis](#43-sweeppy--sensitivity-analysis)
   - 4.4 [main.py — FastAPI Server](#44-mainpy--fastapi-server)
   - 4.5 [test_env.py — Physics Smoke Tests](#45-test_envpy--physics-smoke-tests)
5. [Frontend — `/frontend/`](#5-frontend--frontend)
   - 5.1 [Dashboard.tsx — Main UI Component](#51-dashboardtsx--main-ui-component)
   - 5.2 [FlightScene.tsx — 3D WebGL Visualization](#52-flightscenetsx--3d-webgl-visualization)
   - 5.3 [TelemetryChart.tsx — Plotly Charts](#53-telemetrytsx--plotly-charts)
   - 5.4 [TelemetryTable.tsx — Status Matrix](#54-telemetrytabletsx--status-matrix)
6. [Data Flow — End-to-End](#6-data-flow--end-to-end)
7. [Physics Model Reference](#7-physics-model-reference)
8. [Mission Phase State Machine](#8-mission-phase-state-machine)
9. [Power Management Strategy](#9-power-management-strategy)
10. [API Contract](#10-api-contract)
11. [Key Constants & Tuning Knobs](#11-key-constants--tuning-knobs)
12. [Dependency Map](#12-dependency-map)
13. [Known Constraints & Limitations](#13-known-constraints--limitations)

---

## 1. Project Overview

**AeroOptima** is a propulsion-sizing and mission-simulation tool for a **1000 kg fixed-wing hybrid-electric UAV** built for the IIT Indore × HAL Hackathon (Problem Statement 1).

The system has two loops:

| Loop | What it does | File |
|------|-------------|------|
| **Outer (GA)** | Finds the best `(engine_kw, battery_kwh)` pair that maximises total mission endurance | `optimizer.py` |
| **Inner (Simulation)** | Runs a full Takeoff→Climb→Cruise→Loiter→Descent→Landing mission for one candidate | `environment.py` |

The **frontend** visualises the result in real-time (3D flight path + Plotly charts + telemetry table).

---

## 2. Repository Structure

```
IIT_HAl/
├── backend/
│   ├── environment.py       # Gymnasium env (physics + simulation)
│   ├── optimizer.py         # DEAP GA (propulsion sizing)
│   ├── sweep.py             # Sensitivity sweep (+-10% SFC / battery density)
│   ├── main.py              # FastAPI server (REST API)
│   ├── test_env.py          # Physics smoke tests
│   ├── test_regression.py   # Golden-baseline regression tests
│   └── requirements.txt     # (legacy; use pyproject.toml with uv)
├── data/
│   ├── aerodynamics.json    # Airframe geometry + atmosphere constants
│   ├── turboshaft_specs.json# Engine spec (scaling reference)
│   ├── battery_specs.json   # Battery cell spec + limits
│   └── motor_specs.json     # Fixed EMRAX 228 motor spec
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   └── globals.css
│       └── components/
│           ├── Dashboard.tsx      # Master UI + state management
│           ├── FlightScene.tsx    # Three.js 3D scene
│           ├── TelemetryChart.tsx # Plotly power/resource/trajectory charts
│           └── TelemetryTable.tsx # Scrollable kW status matrix
├── Reports_build/           # Generated PDF/HTML reports (not version-controlled)
├── golden_baseline.json     # Regression baseline snapshot
├── pyproject.toml           # Python package definition (uv-based)
├── uv.lock                  # Locked dependency graph
└── CHANGELOG.md
```

---

## 3. Data Layer — `/data/`

All four JSON config files are loaded once at environment init and drive all physics calculations.

### 3.1 `aerodynamics.json`

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `max_takeoff_weight_kg` | 1000 | kg | Hard MTOW constraint (UAV class limit) |
| `payload_capacity_kg` | 200 | kg | Design payload (used as reference only) |
| `airframe_mass_kg` | 350 | kg | Fixed structural mass |
| `wing_area_m2` | 14.0 | m² | Reference wing area (S) for aerodynamic calcs |
| `wingspan_m` | 15.0 | m | Used to derive aspect ratio AR = b²/S = **16.07** |
| `drag_coefficient_cd0` | 0.025 | — | Parasite drag at zero lift |
| `oswald_efficiency_factor_e` | 0.85 | — | Oswald span efficiency (e) |
| `propeller_efficiency_takeoff` | 0.65 | — | eta_prop for takeoff phase |
| `propeller_efficiency_climb` | 0.75 | — | eta_prop for climb phase |
| `propeller_efficiency_cruise` | 0.85 | — | eta_prop for cruise & loiter |
| `propeller_efficiency_descent` | 0.70 | — | eta_prop for descent & landing |
| `target_cruise_speed_kmh` | 250 | km/h | Reference only (overridden by user input) |
| `target_cruise_altitude_m` | 5000 | m | Reference only |
| `air_density_sea_level_kg_m3` | 1.225 | kg/m³ | ISA sea-level density (rho_0) |
| `gravity_m_s2` | 9.81 | m/s² | Standard gravity (g) |

### 3.2 `turboshaft_specs.json`

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `engine_class` | "Lightweight Turboshaft / APU" | — | Reference class |
| `dry_weight_kg` | 35.0 | kg | Reference dry weight |
| `max_continuous_power_kw` | 60.0 | kW | Reference continuous rating |
| `peak_power_kw` | 75.0 | kW | Reference peak (125% of continuous) |
| `specific_fuel_consumption_kg_per_kwh` | 0.38 | kg/kWh | Nominal SFC at >=80% load (base SFC, `sfc_base`) |
| `fuel_type` | "Jet-A1" | — | Aviation fuel type |
| `fuel_density_kg_per_liter` | 0.8 | kg/L | Used for volumetric reference only |
| `min_size_kw` | 30.0 | kW | GA search lower bound for engine |
| `max_size_kw` | 120.0 | kW | GA search upper bound for engine |
| `reference_power_kw` | 67.5 | kW | Scaling reference for weight (`ref_power`) |
| `reference_weight_kg` | 35.0 | kg | Scaling reference for weight (`ref_weight`) |

**Weight scaling formula:**
`weight_engine = (engine_size_kw / reference_power_kw) × reference_weight_kg`

### 3.3 `battery_specs.json`

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `cell_format` | "21700 Cylindrical" | — | Cell form factor |
| `chemistry` | "Lithium-Ion (NCA)" | — | Cell chemistry |
| `energy_density_wh_per_kg` | 250 | Wh/kg | Pack energy density (used for weight calculation) |
| `max_continuous_discharge_c_rate` | 3.0 | C | `c_rate_continuous` — sustained discharge |
| `max_peak_discharge_c_rate` | 5.0 | C | `c_rate_peak` — short burst limit |
| `charging_efficiency_percent` | 92 | % | Charging efficiency (informational) |
| `min_soc_limit` | 0.10 | fraction | `soc_min` — battery cutoff floor |
| `max_soc_limit` | 0.95 | fraction | Maximum SoC after charge |
| `generator_efficiency` | 0.85 | fraction | Used when motor acts as generator (charging) |
| `min_capacity_kwh` | 16.69 | kWh | GA search lower bound (STANAG 4671 emergency reserve) |
| `max_capacity_kwh` | 50.0 | kWh | GA search upper bound |

**Battery weight formula:**
`weight_battery = (battery_capacity_kwh × 1000) / (energy_density × battery_density_scale)`

### 3.4 `motor_specs.json`

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `manufacturer` | "EMRAX" | — | OEM |
| `model` | "228 Medium Voltage" | — | Fixed off-the-shelf part |
| `mass_kg` | 12.3 | kg | Fixed motor mass (`weight_motor`) |
| `continuous_power_kw` | 55.0 | kW | `motor_continuous_kw` |
| `peak_power_kw` | 109.0 | kW | `motor_peak_kw` |
| `peak_efficiency_percent` | 96 | % | `motor_efficiency` = 0.96 |
| `power_density_kw_per_kg` | 8.86 | kW/kg | Informational |

---

## 4. Backend — `/backend/`

### 4.1 `environment.py` — Gymnasium Simulation

**Class:** `UAVHybridEnv(gym.Env)`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_size_kw` | float | required | Turboshaft shaft power rating to simulate |
| `battery_capacity_kwh` | float | required | Battery pack energy capacity |
| `target_speed_kmh` | float | 250.0 | Target cruise speed (converted to m/s internally) |
| `target_altitude` | float | 5000.0 | Target cruise altitude in metres |
| `payload_weight` | float | 200.0 | Payload mass in kg |
| `data_dir` | str | None | Path to `/data/` folder (auto-resolved if None) |
| `use_heuristic_policy` | bool | True | If True, ignores RL action and applies `_heuristic_psr()` |
| `dt` | float | 10.0 | Simulation timestep in seconds |
| `enable_loiter` | bool | True | Include loiter phase in mission profile |
| `initial_fuel_fraction` | float | 1.0 | Fraction of max fuel capacity loaded (clamped 0.1–1.0) |
| `sfc_scale` | float | 1.0 | Multiplier on SFC (used by sensitivity sweep) |
| `battery_density_scale` | float | 1.0 | Multiplier on energy density (used by sensitivity sweep) |
| `phase_psrs` | dict or None | None | Per-phase PSR override dict (GA PSR mode only) |

#### Derived Instance Attributes

| Attribute | Formula | Description |
|-----------|---------|-------------|
| `target_speed_ms` | `target_speed_kmh / 3.6` | Cruise speed in m/s |
| `aspect_ratio` (AR) | `wingspan² / wing_area` | = 16.07 for default airframe |
| `g` | from JSON | 9.81 m/s² |
| `weight_engine` | `(engine_size_kw / ref_power) × ref_weight` | Scaled engine mass |
| `weight_motor` | `motor_specs["mass_kg"]` | Fixed: 12.3 kg |
| `weight_battery` | `(kwh × 1000) / energy_density` | Scaled battery mass |
| `weight_airframe` | 350 kg (from JSON) | Fixed structural mass |
| `weight_payload` | `payload_weight` param | User-defined |
| `weight_empty_and_payload` | Sum of all above | Total non-fuel mass |
| `mtow` | 1000 kg (from JSON) | Maximum take-off weight |
| `fuel_initial` | `(mtow - empty_and_payload) × fuel_fraction` | Initial fuel load |
| `motor_continuous_kw` | 55.0 kW | From motor spec |
| `motor_peak_kw` | 109.0 kW | From motor spec |
| `motor_efficiency` | 0.96 | From motor spec |
| `engine_continuous_kw` | `min(engine_size_kw, spec_continuous × scale)` | Altitude-derated base |
| `engine_peak_kw` | `min(engine_size_kw × 1.25, spec_peak × scale)` | 125% rating |
| `sfc_base` | `spec_sfc × sfc_scale` | Base specific fuel consumption |
| `c_rate_continuous` | 3.0 | From battery spec |
| `c_rate_peak` | 5.0 | From battery spec |
| `soc_min` | 0.10 | From battery spec |

#### Mission Phase Constants

| Constant | Value |
|----------|-------|
| `PHASE_TAKEOFF` | `"takeoff"` |
| `PHASE_CLIMB` | `"climb"` |
| `PHASE_CRUISE` | `"cruise"` |
| `PHASE_LOITER` | `"loiter"` |
| `PHASE_DESCENT` | `"descent"` |
| `PHASE_LANDING` | `"landing"` |
| `PHASE_COMPLETED` | `"completed"` |

#### Gymnasium Spaces

| Space | Low | High | Shape | dtype |
|-------|-----|------|-------|-------|
| `observation_space` (Box) | `[0, 0, 0, 0, 0]` | `[12000, 150, 1, 500, 500]` | (5,) | float32 |
| `action_space` (Box) | `[-1.0]` | `[1.0]` | (1,) | float32 |

**Observation vector:** `[altitude_m, speed_m/s, soc_0-1, fuel_remaining_kg, p_required_kW]`
**Action:** Power Split Ratio (PSR) where `-1.0` = 100% generator (charging), `0.0` = 100% engine, `+1.0` = 100% motor.

#### Mutable State Variables (reset at `env.reset()`)

| Variable | Initial value | Description |
|----------|--------------|-------------|
| `altitude` | 0.0 | Current altitude (m) |
| `speed` | 0.0 | Current TAS (m/s) |
| `soc` | `battery_specs["max_soc_limit"]` = 0.95 | State of charge (fraction) |
| `fuel_remaining` | `fuel_initial` | Remaining fuel mass (kg) |
| `time_elapsed` | 0.0 | Elapsed simulation time (s) |
| `current_phase` | `"takeoff"` | Current mission phase |
| `deficit_counter` | 0 | Consecutive power-deficit steps |
| `power_deficit_flag` | False | True if achievable climb < target |
| `climb_prevented` | False | True if zero climb rate in climb phase |
| `flight_log` | `[]` | List of telemetry dicts (one per step) |

#### Telemetry Log Entry Schema (per step)

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Elapsed time (s), rounded to 2 dp |
| `altitude` | float | Altitude (m), 1 dp |
| `speed` | float | TAS (m/s), 2 dp |
| `power_required` | float | P_req (kW), 3 dp |
| `power_delivered` | float | P_delivered (kW), 3 dp |
| `power_motor` | float | Electric motor power (kW), 3 dp |
| `power_engine` | float | Engine shaft power (kW), 3 dp |
| `soc` | float | Battery SoC fraction, 5 dp |
| `fuel` | float | Fuel remaining (kg), 4 dp |
| `weight` | float | Total aircraft weight (kg), 2 dp |
| `phase` | str | Mission phase string |
| `deficit` | float | Unmet power (kW), 3 dp |
| `u` | float | Actual PSR = p_motor / p_req (0–1), 4 dp |
| `p_aero` | float | Aerodynamic drag power (kW), 3 dp |
| `p_climb` | float | Climb rate power (kW), 3 dp |
| `climb_rate` | float | Vertical speed (m/s), 3 dp |

#### Key Private Methods

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| `_load_constants` | `()` | None | Loads all 4 JSON files into `self.aero`, `self.engine_specs`, `self.battery_specs`, `self.motor_specs` |
| `_atmosphere` | `(altitude_m: float)` | float (rho kg/m³) | ISA standard atmosphere; troposphere below 11 km, stratosphere above |
| `_stall_speed` | `(weight, rho, cl_max=1.5)` | float (m/s) | V_stall = sqrt(2·W·g / rho·S·CL_max) |
| `_prop_efficiency` | `(phase: str)` | float | Look up propeller efficiency by mission phase |
| `_compute_power_required` | `(weight, speed, altitude, climb_rate, phase)` | `(p_req_kw, p_aero_kw, p_climb_kw)` | Drag polar + Oswald + climb power |
| `_effective_sfc` | `(engine_power_kw)` | float (kg/kWh) | SFC with partial-load penalty |
| `_heuristic_psr` | `(phase, soc, fuel_ratio)` | float (-1 to 1) | Zhang-inspired PSR policy |
| `_max_battery_power` | `(is_peak=False)` | float (kW) | min(C-rate limit, motor rating) |
| `_max_engine_power` | `(altitude_m, is_peak=False)` | float (kW) | Gagg-Ferrar altitude derated engine power |
| `_log_telemetry` | `(psr, p_req, p_motor, p_engine, p_delivered, p_deficit, p_aero, p_climb, climb_rate)` | None | Appends one dict to `flight_log` |
| `_get_obs` | `(power_required)` | np.ndarray (5,) | Returns observation vector |

#### Step Return Schema

| Return | Type | Description |
|--------|------|-------------|
| `obs` | np.ndarray (5,) | `[alt, speed, soc, fuel, p_req]` |
| `reward` | float | +1 base, +2 cruise, +2.5 loiter, +500 mission complete, -500 stall, -p_deficit×10 |
| `terminated` | bool | True on mission complete, stall, out of energy, or 3x power deficit |
| `truncated` | bool | True at 30-hour safety limit (108 000 s) |
| `info` | dict | `{"reason": str}` — termination reason string |

#### Termination Conditions (priority order)

1. `fuel_initial <= 0` → immediate -1000 reward, `terminated=True`
2. `CL_actual > 1.6` → stall, `terminated=True`
3. `current_phase == "completed"` → success, `terminated=True`
4. `soc <= 0.01 AND fuel_remaining <= 0.01` → energy depleted, `terminated=True`
5. `deficit_counter >= 3 AND p_deficit > 1.0 kW` → sustained power deficit, `terminated=True`
6. `time_elapsed >= 108 000 s` → safety truncation, `truncated=True`

---

### 4.2 `optimizer.py` — DEAP Genetic Algorithm

#### Function: `load_bounds(data_dir)`

| Return key | Type | Source |
|------------|------|--------|
| `"engine"` | `(min_kw, max_kw)` | turboshaft_specs.json → (30.0, 120.0) |
| `"battery"` | `(min_kwh, max_kwh)` | battery_specs.json → (16.69, 50.0) |

#### Function: `evaluate_individual(individual, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `individual` | list | required | `[engine_kw, battery_kwh]` or `[engine_kw, battery_kwh, psr_to, psr_cl, psr_cr, psr_lo]` |
| `target_speed_kmh` | float | — | Passed through to `UAVHybridEnv` |
| `target_altitude` | float | — | Passed through to `UAVHybridEnv` |
| `payload_weight` | float | — | Passed through to `UAVHybridEnv` |
| `data_dir` | str | — | Path to `/data/` |
| `bounds` | dict | — | Clipping bounds for design variables |
| `enable_loiter` | bool | True | Passed through to `UAVHybridEnv` |
| `initial_fuel_fraction` | float | 1.0 | Passed through to `UAVHybridEnv` |
| `optimize_psr` | bool | False | If True, individual includes 4 PSR values (6-gene mode) |

**Returns:** `(endurance_hours,)` — single-objective fitness tuple. Multiplied by **0.4** penalty if mission does not land successfully.

#### Function: `optimize_propulsion(...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_speed_kmh` | float | 250.0 | Mission cruise speed |
| `target_altitude` | float | 5000.0 | Mission cruise altitude (m) |
| `payload_weight` | float | 200.0 | Payload mass (kg) |
| `data_dir` | str | None | Auto-resolved if None |
| `pop_size` | int | 40 | GA population size |
| `n_gen` | int | 15 | Number of GA generations |
| `enable_loiter` | bool | True | Enable loiter phase |
| `initial_fuel_fraction` | float | 1.0 | Initial fuel fill fraction |
| `optimize_psr` | bool | False | 6-gene mode (also optimises PSR policies) |

**DEAP Configuration:**

| Setting | Value | Notes |
|---------|-------|-------|
| Fitness direction | Maximise (weight=+1.0) | Maximise endurance |
| Crossover | `cxBlend` (alpha=0.5) | Blend crossover |
| Mutation | `mutGaussian` (sigma=[8.0, 4.0]) | sigma=[8,4,0.15,0.15,0.15,0.15] in PSR mode |
| Mutation probability (`indpb`) | 0.35 | Per-gene |
| Selection | Tournament (tournsize=3) | |
| Crossover prob (`CXPB`) | 0.6 | Per-pair |
| Mutation prob (`MUTPB`) | 0.35 | Per-individual |
| Hall of Fame size | 1 | Best individual only |

**Return dict:**
```python
{
  "engine_size_kw":       float,
  "battery_capacity_kwh": float,
  "fitness":              float,   # expected endurance (hours)
  # Only present when optimize_psr=True:
  "phase_psrs": {
    "takeoff": float,
    "climb":   float,
    "cruise":  float,
    "loiter":  float,
  }
}
```

---

### 4.3 `sweep.py` — Sensitivity Analysis

#### Function: `run_sensitivity_sweep(...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_kw` | float | required | Optimal engine size from GA |
| `battery_kwh` | float | required | Optimal battery size from GA |
| `target_speed_kmh` | float | 250.0 | Mission cruise speed |
| `target_altitude` | float | 5000.0 | Mission cruise altitude |
| `payload_weight` | float | 200.0 | Payload mass |
| `enable_loiter` | bool | True | Enable loiter |
| `initial_fuel_fraction` | float | 1.0 | Fuel fill fraction |
| `data_dir` | str | None | Data directory path |

**Scenarios (4 runs + 1 nominal = 5 total simulations):**

| Scenario | `sfc_scale` | `battery_density_scale` |
|----------|-------------|------------------------|
| Nominal | 1.0 | 1.0 |
| +10% SFC, +10% density | 1.1 | 1.1 |
| +10% SFC, -10% density | 1.1 | 0.9 |
| -10% SFC, +10% density | 0.9 | 1.1 |
| -10% SFC, -10% density | 0.9 | 0.9 |

**Return dict:**
```python
{
  "nominal":       float,   # Nominal endurance (hours)
  "cases": [
    {"name": str, "sfc_scale": float, "battery_density_scale": float, "endurance_hours": float},
    ...
  ],
  "min_endurance": float,   # Worst-case endurance across 4 scenarios
  "max_endurance": float,   # Best-case endurance across 4 scenarios
}
```

---

### 4.4 `main.py` — FastAPI Server

**App:** `FastAPI(title="AeroOptima...", version="2.0.0")`
**Host/Port:** `127.0.0.1:8000` (dev; `uvicorn main:app --reload`)
**CORS Origins:** `localhost:3000`, `127.0.0.1:3000`, `localhost:8000`, `127.0.0.1:8000`

#### Pydantic Request Model — `OptimizationRequest`

| Field | Type | Default | Validation | Description |
|-------|------|---------|------------|-------------|
| `target_speed_kmh` | float | 250.0 | ge=100, le=400 | Cruise speed (km/h) |
| `target_altitude` | float | 5000.0 | ge=500, le=10000 | Cruise altitude (m) |
| `payload_weight` | float | 200.0 | ge=50, le=350 | Payload mass (kg) |
| `enable_loiter` | bool | True | — | Toggle loiter phase |
| `initial_fuel_fraction` | float | 1.0 | ge=0.1, le=1.0 | Fuel fill fraction |

#### Pydantic Response Model — `OptimalSpecs`

| Field | Type | Description |
|-------|------|-------------|
| `engine_kw` | float | Optimal engine power (kW) |
| `battery_kwh` | float | Optimal battery capacity (kWh) |
| `motor_kw` | float | Fixed EMRAX peak power (109 kW) |
| `endurance_hours` | float | Total mission endurance (h) |
| `endurance_hours_min` | float or None | Worst-case from sensitivity sweep |
| `endurance_hours_max` | float or None | Best-case from sensitivity sweep |
| `empty_weight_kg` | float | Structural + propulsion mass (excl. fuel) |
| `fuel_weight_kg` | float | Initial fuel load (kg) |
| `total_weight_kg` | float | MTOW (always <= 1000 kg) |
| `motor_model` | str | "EMRAX 228 Medium Voltage" |
| `engine_weight_kg` | float | Engine mass (kg) |
| `motor_weight_kg` | float | Motor mass (12.3 kg) |
| `battery_weight_kg` | float | Battery pack mass (kg) |

#### Pydantic Response Model — `TelemetryPoint`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time` | float | — | Elapsed time (s) |
| `altitude` | float | — | Altitude (m) |
| `speed` | float | — | TAS (m/s) |
| `power_required` | float | — | P_req (kW) |
| `power_delivered` | float | — | P_delivered (kW) |
| `power_motor` | float | — | Electric motor power (kW) |
| `power_engine` | float | — | Engine power (kW) |
| `soc` | float | — | State of charge (0–1) |
| `fuel` | float | — | Remaining fuel (kg) |
| `weight` | float | — | Total weight (kg) |
| `phase` | str | — | Mission phase string |
| `deficit` | float | — | Power deficit (kW) |
| `u` | float | — | Actual PSR |
| `sfc` | float | 0.38 | Effective SFC at this step (kg/kWh) |
| `p_aero` | float | 0.0 | Aerodynamic power (kW) |
| `p_climb` | float | 0.0 | Climb power (kW) |
| `climb_rate` | float | 0.0 | Vertical speed (m/s) |

#### Endpoints

| Method | Path | Body | Response | Description |
|--------|------|------|----------|-------------|
| `POST` | `/api/optimize` | `OptimizationRequest` | `OptimizationResponse` | Full GA + simulation pipeline |
| `GET` | `/api/health` | — | `{"status": "healthy", "motor": str}` | Health check |

**Processing pipeline for `POST /api/optimize`:**
1. Call `optimize_propulsion()` → GA result `(engine_kw, battery_kwh)`
2. Re-simulate with `UAVHybridEnv(dt=60s, use_heuristic_policy=True)` for clean 1-min telemetry
3. Call `run_sensitivity_sweep()` → endurance range
4. Compute `effective_sfc` per telemetry step
5. Return `OptimizationResponse(optimal_specs, telemetry)`

---

### 4.5 `test_env.py` — Physics Smoke Tests

Two test functions (run directly, not via pytest):

| Function | What it validates |
|----------|-----------------|
| `test()` | Weight budget, altitude derating at 0/3000/5000 m, full simulation run |
| `test_step2_phantom_climb_fixes()` | Case A: adequate power → altitude increases; Case B: undersized power → no phantom climb |

---

## 5. Frontend — `/frontend/`

**Stack:** Next.js 16.2 + React 19 + TypeScript + TailwindCSS 4 + Three.js + Plotly

### 5.1 `Dashboard.tsx` — Main UI Component

**React State Variables:**

| State | Type | Default | Description |
|-------|------|---------|-------------|
| `targetSpeedKmh` | number | 250 | Cruise speed slider (150–350 km/h, step 5) |
| `targetAltitude` | number | 5000 | Altitude slider (3000–10000 m, step 250) |
| `payloadWeight` | number | 200 | Payload slider (100–300 kg, step 5) |
| `enableLoiter` | boolean | true | Loiter phase checkbox |
| `showMatrix` | boolean | true | Toggle Propulsion Status Matrix panel |
| `initialFuelFraction` | number | 1.0 | Fuel fraction slider (0.1–1.0, step 0.05) |
| `loading` | boolean | false | API request in progress |
| `error` | string or null | null | Error message from backend |
| `specs` | OptimalSpecs or null | null | Backend optimal specs result |
| `telemetry` | TelemetryPoint[] | [] | Full telemetry time series |
| `currentIndex` | number | 0 | Scrubber position (index into telemetry) |
| `isPlaying` | boolean | false | Auto-play playback |
| `activeTab` | '3d' or 'charts' | '3d' | Tab selection |
| `genCount` | number | 1 | Cosmetic GA generation counter (1–15) |
| `loadProgress` | number | 100 | Cosmetic progress bar (0–100%) |

**TypeScript Interfaces:**

```typescript
interface OptimalSpecs {
  engine_kw: number;
  battery_kwh: number;
  motor_kw: number;
  endurance_hours: number;
  endurance_hours_min?: number;
  endurance_hours_max?: number;
  empty_weight_kg: number;
  fuel_weight_kg: number;
  total_weight_kg: number;
  motor_model: string;
  engine_weight_kg: number;
  motor_weight_kg: number;
  battery_weight_kg: number;
}

interface TelemetryPoint {
  time: number;
  altitude: number;
  speed: number;
  power_required: number;
  power_delivered: number;
  power_motor: number;
  power_engine: number;
  soc: number;
  fuel: number;
  weight: number;
  phase: string;
  deficit: number;
  u: number;
  p_aero: number;
  p_climb: number;
  climb_rate: number;
  sfc: number;
}
```

**Computed (useMemo):**

| Variable | Description |
|----------|-------------|
| `currentPoint` | `telemetry[currentIndex]` — live data point |
| `phaseDurations` | `{ [phase]: { start, end } }` — time ranges per phase |
| `totalMissionTime` | Sum of all phase durations (seconds) |
| `weightBreakdown` | Array of `{ name, weight, color }` for MTOW chart |
| `totalWeight` | Sum of all weight breakdown entries |

**Weight Breakdown Hardcoded Colors:**

| Component | Color |
|-----------|-------|
| Airframe (350 kg fixed) | `#4B5563` |
| Payload | `#6366F1` |
| Turboshaft | `#3B82F6` |
| EMRAX Motor | `#10B981` |
| Battery | `#F59E0B` |
| Fuel (Jet-A1) | `#EF4444` |

**Phase badge CSS classes (`PHASE_BADGE`):**

| Phase | Tailwind classes |
|-------|-----------------|
| `takeoff` | `bg-red-900/60 text-red-300 border-red-700/40` |
| `climb` | `bg-amber-900/60 text-[#FFB454] border-amber-700/40` |
| `cruise` | `bg-cyan-900/60 text-[#E8EDF2] border-cyan-700/40` |
| `loiter` | `bg-violet-900/60 text-violet-300 border-violet-700/40` |
| `descent` | `bg-teal-900/60 text-teal-300 border-teal-700/40` |
| `landing` | `bg-emerald-900/60 text-[#FFB454] border-[#1F2733]` |
| `completed` | `bg-slate-800/60 text-[#5C6773] border-[#1F2733]` |

**Design tokens (CSS color system):**

| Token | Value | Usage |
|-------|-------|-------|
| `#0A0E14` | Near-black | App background |
| `#12161F` | Dark panel | Card/cell backgrounds |
| `#1F2733` | Dark border | Panel borders |
| `#E8EDF2` | Off-white | Primary text |
| `#5C6773` | Muted gray | Label/secondary text |
| `#FFB454` | Amber-gold | Accent / HAL brand color |

**Environment variable:**
- `NEXT_PUBLIC_API_URL` — backend URL override (falls back to `http://{hostname}:8000`)

**Auto-run behaviour:** `useEffect` triggers `handleOptimize()` immediately on mount with `setTimeout(..., 0)`.

---

### 5.2 `FlightScene.tsx` — 3D WebGL Visualization

**Library:** `@react-three/fiber` + `@react-three/drei` + Three.js
**Camera:** OrthographicCamera (top-down/side view with OrbitControls)

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `telemetry` | TelemetryPoint[] | Full telemetry series |
| `currentIndex` | number | Currently highlighted point |

**Scale constants:**

| Constant | Value | Effect |
|----------|-------|--------|
| `ALT_SCALE` | 0.003 | 5000 m → 15 scene units |
| `DIST_SCALE` | 0.001 | Horizontal distance scaling |

**Flight path generation (`generateFlightPath`):**
- Accumulates horizontal distance from speed × cos(climb_angle) × dt
- Loiter phase: sinusoidal racetrack orbit in X/Z plane centred at loiter start point
- Descent/Landing phase: lerps X back toward origin (return-to-base simulation)

**Colour coding by power split (u field):**

| Condition | Colour | Mode |
|-----------|--------|------|
| `u > 0.4` | Amber (#f59e0b) | Battery-dominant |
| `0.05 < u <= 0.4` | Cyan (#22d3ee) | Hybrid |
| `u <= 0.05` | Emerald (#10b981) | Engine-dominant |
| (no thrust/idle) | Slate (#64748b) | Idle |

---

### 5.3 `TelemetryChart.tsx` — Plotly Charts

**Library:** `react-plotly.js` (dynamically imported, SSR disabled)

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `telemetry` | TelemetryPoint[] | Full telemetry series |

**Tabs:**

| Tab | Plots |
|-----|-------|
| `power` | Power required vs delivered, motor vs engine breakdown, electric fraction (PSR), power deficit |
| `resources` | Battery SoC (%), fuel remaining (kg), aircraft weight (kg) |
| `trajectory` | Altitude (m), airspeed (km/h) vs mission time |

**Summary stats (top row):**

| Stat | Calculation |
|------|-------------|
| Max Altitude | `max(telemetry[*].altitude)` |
| Total Endurance | `telemetry[-1].time / 3600` |
| Fuel Burned | `telemetry[0].fuel - telemetry[-1].fuel` |
| Peak Power | `max(telemetry[*].power_required)` |

---

### 5.4 `TelemetryTable.tsx` — Status Matrix

**Purpose:** Compact tabular view of all telemetry (downsampled to max 120 rows), clickable rows scrub the 3D view.

**Props:**

| Prop | Type | Description |
|------|------|-------------|
| `telemetry` | TelemetryPoint[] | Full telemetry series |
| `currentIndex` | number | Currently highlighted row |
| `onIndexChange` | `(index: number) => void` | Callback to update scrubber |

**`downsampleIndices(total, maxRows=120)`:** Evenly spaced index selection for large telemetry arrays.

**`fuelColor(fuel, maxFuel)`:** Linearly interpolates hex color green (#10b981) → red (#ef4444) as fuel depletes.

**Columns displayed:**

| Column | Source field | Description |
|--------|-------------|-------------|
| `T` | `time` | Mission time (mm:ss) |
| `Phase` | `phase` | Phase badge |
| `Alt` | `altitude` | Altitude (m) |
| `TAS` | `speed × 3.6` | km/h |
| `P_req` | `power_required` | Required power (kW) |
| `Motor` | `power_motor` | Electric power (kW) |
| `Engine` | `power_engine` | Engine shaft power (kW) |
| `SoC` | `soc × 100` | Battery % |
| `Fuel` | `fuel` | Remaining fuel kg (green to red gradient) |
| `CR` | `climb_rate` | m/s |

---

## 6. Data Flow — End-to-End

```
User (browser sliders)
       |
       |  POST /api/optimize
       |  { target_speed_kmh, target_altitude, payload_weight,
       |    enable_loiter, initial_fuel_fraction }
       v
main.py → optimize_propulsion()
       |
       |  GA outer loop (40 pop × 15 gen)
       |  Each individual = [engine_kw, battery_kwh]
       |
       +---► evaluate_individual()
       |         └───► UAVHybridEnv(dt=60s, heuristic=True)
       |               Full mission simulation → endurance (hours)
       |
       |  Hall of Fame → best (engine_kw, battery_kwh)
       |
       +---► Re-simulate best individual (dt=60s, heuristic=True)
       |         → flight_log (1 entry/min)
       |
       +---► run_sensitivity_sweep() → min/max endurance range
       |
       └───► Build OptimizationResponse
               { optimal_specs, telemetry[] }
                       |
                       |  JSON response
                       v
Dashboard.tsx
       +---► setSpecs(optimal_specs)
       +---► setTelemetry(telemetry)
       +---► FlightScene ← 3D path from telemetry
       +---► TelemetryChart ← Plotly time series
       └───► TelemetryTable ← Scrollable matrix
```

---

## 7. Physics Model Reference

### 7.1 Atmosphere — ISA Standard

```
Troposphere (h < 11 000 m):
  rho(h) = rho_0 * (1 - 2.25577e-5 * h)^4.25588

Stratosphere (h >= 11 000 m):
  rho(h) = 0.3639 * exp(-(h - 11 000) / 6341.6)

Clamp: rho >= 0.05 kg/m³
```

### 7.2 Aerodynamics — Oswald Drag Polar

```
AR = wingspan² / wing_area = 15² / 14 ≈ 16.07

gamma  = arcsin(climb_rate / speed)                [flight path angle]
CL     = 2*W*g*cos(gamma) / (rho*V²*S)            [lift coefficient]
CD     = CD0 + CL² / (pi*AR*e)                    [Oswald drag polar]

P_aero  = 0.5*rho*V³*S*CD                          [aerodynamic drag power, W]
P_climb = W*g*Vz                                   [climb power, W]
P_shaft = (P_aero + P_climb) / eta_prop            [shaft power required, W]
```

### 7.3 Engine — Gagg-Ferrar Altitude Derating

```
sigma = rho(h) / rho_0
P_max_ice(h) = P_max_ice_SL * (sigma - (1-sigma)/7.55)
Clamp: P_max_ice >= 0
```

### 7.4 SFC Partial-Load Penalty

```
load_fraction = engine_power_kw / engine_continuous_kw

load >= 0.8:       SFC = sfc_base             (optimal)
0.5 <= load < 0.8: SFC = sfc_base * (1 + 0.15 * (0.8 - load) / 0.3)  (up to +15%)
load < 0.5:        SFC = sfc_base * (1.15 + 0.25 * (0.5 - load) / 0.5) (up to +40%)
```

### 7.5 Resource Consumption

```
Fuel burn:    dm = SFC(load) * P_engine * dt_hours            [kg]
Battery draw: p_batt = p_motor / motor_efficiency             (discharging, motor_eff=0.96)
              p_batt = p_motor * generator_efficiency         (charging, gen_eff=0.85)
              dSoC   = (p_batt * dt_hours) / battery_capacity_kwh
```

### 7.6 Excess-Power Climb Limit ("Phantom Climb Fix")

Prevents altitude from increasing when available thrust power < power needed to sustain target climb rate:

```
p_excess_thrust = (p_engine_avail + p_elec_avail) * eta_prop - P_aero_level_flight
Vz_actual       = max(0, p_excess_thrust * 1000 / (W * g))
climb_rate      = min(climb_rate_target, Vz_actual)
```

---

## 8. Mission Phase State Machine

```
TAKEOFF  --(alt >= 200 m)--> CLIMB
CLIMB    --(alt >= target_altitude)--> CRUISE
CRUISE   --(loiter=True, fuel_ratio < 0.40)--> LOITER
         --(loiter=False, fuel_ratio < 0.08 & soc < 0.15 OR fuel_ratio < 0.03)--> DESCENT
LOITER   --(fuel_ratio < 0.08 & soc < 0.15 OR fuel_ratio < 0.03)--> DESCENT
DESCENT  --(alt <= 200 m)--> LANDING
LANDING  --(alt <= 5 m)--> COMPLETED
```

**Speed setpoints per phase:**

| Phase | Speed setpoint | Climb Rate target |
|-------|---------------|-------------------|
| Takeoff | max(1.15 × V_stall, 25 m/s) | +3.0 m/s |
| Climb | max(1.3 × V_stall, 35 m/s) | +5.0 m/s |
| Cruise | max(target_speed_ms, 1.25 × V_stall) | 0 m/s |
| Loiter | max(0.76 × target_speed_ms, 1.2 × V_stall) | 0 m/s |
| Descent | max(1.2 × V_stall, 30 m/s) | -1.5 m/s |
| Landing | max(1.1 × V_stall, 22 m/s) | -0.8 m/s |

---

## 9. Power Management Strategy

Based on **Zhang et al.** hybrid power management (heuristic policy):

| Phase | Base PSR | Condition overrides |
|-------|----------|-------------------|
| Takeoff | 0.65 | soc < 0.3 → 0.30 |
| Climb | 0.50 | soc < 0.3 → 0.20; soc < 0.5 → 0.35 |
| Cruise | 0.12 | soc > 0.8 → 0.20; soc < 0.3 → 0.00 |
| Loiter | 0.00 | fuel_ratio < 0.1 AND soc > 0.3 → 0.50 |
| Descent | 0.00 | — |
| Landing | 0.00 | — |

**PSR semantics:**
- `PSR = 0.0` → 100% engine
- `PSR = 1.0` → 100% electric motor
- `PSR < 0.0` → engine drives both propeller AND generator (battery charging mode)

**Power split constraint enforcement order:**
1. Clip motor by SoC floor (`soc_min = 0.10`)
2. Clip motor by C-rate / motor rating (`_max_battery_power`)
3. Clip engine by Gagg-Ferrar altitude derated max (`_max_engine_power`)
4. Active load-sharing: if deficit remains, try extra from motor, then engine
5. Remaining deficit reported in `p_deficit` field

---

## 10. API Contract

### POST `/api/optimize`

```
Request:
  Content-Type: application/json
  {
    "target_speed_kmh":      float   (100-400, default 250),
    "target_altitude":       float   (500-10000, default 5000),
    "payload_weight":        float   (50-350, default 200),
    "enable_loiter":         boolean (default true),
    "initial_fuel_fraction": float   (0.1-1.0, default 1.0)
  }

Response 200:
  {
    "optimal_specs": {
      "engine_kw":            float,
      "battery_kwh":          float,
      "motor_kw":             float,
      "endurance_hours":      float,
      "endurance_hours_min":  float | null,
      "endurance_hours_max":  float | null,
      "empty_weight_kg":      float,
      "fuel_weight_kg":       float,
      "total_weight_kg":      float,
      "motor_model":          string,
      "engine_weight_kg":     float,
      "motor_weight_kg":      float,
      "battery_weight_kg":    float
    },
    "telemetry": [
      {
        "time":             float,
        "altitude":         float,
        "speed":            float,
        "power_required":   float,
        "power_delivered":  float,
        "power_motor":      float,
        "power_engine":     float,
        "soc":              float,
        "fuel":             float,
        "weight":           float,
        "phase":            string,
        "deficit":          float,
        "u":                float,
        "sfc":              float,
        "p_aero":           float,
        "p_climb":          float,
        "climb_rate":       float
      },
      ...
    ]
  }

Response 500:
  { "detail": "Optimization failed: <error message>" }
```

### GET `/api/health`

```
Response 200:
  { "status": "healthy", "motor": "228 Medium Voltage" }
```

---

## 11. Key Constants & Tuning Knobs

| Constant | Location | Value | Impact |
|----------|----------|-------|--------|
| `CXPB` (crossover prob) | optimizer.py | 0.6 | GA exploration/exploitation |
| `MUTPB` (mutation prob) | optimizer.py | 0.35 | GA diversity |
| `pop_size` | optimizer.py | 40 | GA population |
| `n_gen` | optimizer.py | 15 | GA generations |
| `tournsize` | optimizer.py | 3 | Selection pressure |
| Climb rate target (takeoff) | environment.py | 3.0 m/s | Takeoff performance |
| Climb rate target (climb) | environment.py | 5.0 m/s | Cruise altitude reach rate |
| Idle descent rate | environment.py | -1.5 m/s | Descent duration |
| Final approach rate | environment.py | -0.8 m/s | Landing duration |
| Loiter trigger fuel_ratio | environment.py | < 0.40 | When to start loiter |
| Descent trigger fuel_ratio | environment.py | < 0.08 + soc < 0.15 | When to descend |
| Stall CL limit | environment.py | 1.6 | Termination threshold |
| 30h safety limit | environment.py | 108 000 s | Simulation truncation |
| GA dt (evaluation) | optimizer.py | 60 s | Speed vs accuracy |
| Telemetry dt (API) | main.py | 60 s | Response size |
| Auto-play frame rate | Dashboard.tsx | 30 ms/frame | Playback speed |
| Table downsample | TelemetryTable.tsx | 120 rows | Display performance |

---

## 12. Dependency Map

### Python (backend)

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `fastapi` | >=0.100.0 | REST API framework |
| `uvicorn` | >=0.22.0 | ASGI server |
| `pydantic` | >=2.0 | Request/response validation |
| `deap` | >=1.3.3 | Genetic algorithm framework |
| `gymnasium` | >=0.28.1 | RL environment base class |
| `numpy` | >=1.24.0 | Array operations |
| `scipy` | >=1.10.0 | Scientific computing (available, minor usage) |
| `pytest` | >=7.0.0 | Testing |

### JavaScript (frontend)

| Package | Version | Purpose |
|---------|---------|---------|
| `next` | 16.2.10 | React framework (SSR + routing) |
| `react` | 19.2.4 | UI library |
| `three` | ^0.185.1 | 3D WebGL engine |
| `@react-three/fiber` | ^9.6.1 | React bindings for Three.js |
| `@react-three/drei` | ^10.7.7 | Three.js helpers (Line, Grid, OrbitControls, etc.) |
| `react-plotly.js` | ^4.0.0 | Plotly React wrapper |
| `plotly.js-dist-min` | ^3.7.0 | Plotly core (minified) |
| `tailwindcss` | ^4 | Utility CSS |
| `typescript` | ^5 | Type safety |

---

## 13. Known Constraints & Limitations

| Area | Constraint | Notes |
|------|-----------|-------|
| **MTOW** | Hard 1000 kg limit | Enforced by `fuel_initial <= 0` check |
| **Motor** | Fixed EMRAX 228 | Not a design variable; weight always 12.3 kg |
| **Engine range** | 30–120 kW | GA search bounds from JSON |
| **Battery range** | 16.69–50 kWh | GA search bounds (min set by STANAG 4671) |
| **Atmosphere** | ISA only | No wind, no turbulence, no humidity |
| **Flight path** | 2D (altitude vs distance) | Lateral nav simplified; loiter is sinusoidal approximation |
| **SFC model** | Partial-load penalty only | No temperature effect, no altitude effect on SFC |
| **Battery** | No thermal model | C-rate and SoC limits only |
| **GA** | Single-objective | Only endurance; no multi-objective Pareto front |
| **Simulation** | Sequential (no parallelism) | GA can be slow (~90 s for 40×15 runs) |
| **PSR optimisation** | Optional (`--optimize-psr` flag) | Not used in default API endpoint |
| **Frontend** | No auth/persistence | State resets on page refresh |
| **CORS** | localhost only | Needs update for cloud deployment |

---

*Report generated: 2026-08-07 | Codebase: d:/HAL/IIT_HAl | Stack: FastAPI + DEAP + Gymnasium (backend) | Next.js 16 + Three.js + Plotly (frontend)*
