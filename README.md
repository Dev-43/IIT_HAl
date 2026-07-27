# AeroOptima — Hybrid-Electric UAV Propulsion Optimization Platform
### IIT Indore × HAL Hackathon (Problem Statement PS-1)

AeroOptima is an AI-driven sizing and simulation platform designed to optimize the hybrid-electric propulsion architecture for a **1000 kg tactical fixed-wing UAV**. The platform couples a genetic algorithm (outer optimization loop) with a high-fidelity flight physics environment (inner loop) to size the engine power (kW) and battery capacity (kWh) for maximum mission endurance.

---

## 🏗️ Project Architecture

The workspace is organized into three main directory layers:

```
HAL/ (Root)
├── README.md                  # This file (Project Overview & Setup)
├── data/                      # UAV Component Specification Database
│   ├── aerodynamics.json      # Lift, drag, wingspan, and atmospheric constants
│   ├── battery_specs.json     # Li-Ion density, C-rate, and SoC limits
│   ├── motor_specs.json       # EMRAX motor ratings and weight
│   └── turboshaft_specs.json  # Reference turboshaft ratings and SFC curves
├── backend/                   # Python FastAPI & Physics Simulator
│   ├── environment.py         # Custom Gymnasium flight environment
│   ├── optimizer.py           # Genetic Algorithm sizing loop (DEAP)
│   └── main.py                # REST API and flight re-simulator
└── frontend/                  # Next.js 16 Web Dashboard
    └── src/
        ├── components/
        │   ├── FlightScene.tsx     # 3D Three.js tactical flight visualizer
        │   ├── TelemetryChart.tsx  # Plotly.js unified telemetry charts
        │   ├── TelemetryTable.tsx  # Scrollable propulsion status matrix
        │   └── Dashboard.tsx       # State management & parameter controls
        └── app/
            └── page.tsx            # Main page entrypoint
```

---

## ⚡ Running Locally

To start the full-stack application from a fresh checkout, open two separate terminal sessions in the project root directory:

### 1. Start the Backend (FastAPI)
Verify python dependencies are synced and start the Uvicorn server:
```powershell
# In terminal 1 (project root directory):
uv run python backend/main.py
```
*The FastAPI backend will start running locally at **`http://127.0.0.1:8000`**.*

### 2. Start the Frontend (Next.js)
Install node modules and start the Next.js development server:
```powershell
# In terminal 2 (project root directory):
cd frontend
npm install
npm run dev
```
*The web dashboard will start running locally at **`http://localhost:3000`**.*

---

## ⚙️Sizing & Optimization Framework

1. **Outer Optimization Loop (DEAP Genetic Algorithm):** Sizes two continuous parameters:
   * **Engine shaft rating:** $30\text{ kW} \le P_{engine} \le 120\text{ kW}$
   * **Battery pack capacity:** $5\text{ kWh} \le E_{batt} \le 50\text{ kWh}$
2. **Inner Simulation Loop (Gymnasium Environment):** Simulates a full 6-phase mission profile (Takeoff, Climb, Cruise, Loiter, Descent, Landing) using standard ISA atmosphere and Oswald aerodynamic drag models.
3. **Power-Split Strategy:** Governed by an intelligent heuristic rule (Zhang et al.) prioritizing the motor for high-torque phases (takeoff/climb) and the engine at optimal load factors during cruise.

---

## 🔒 Military Airworthiness Compliance (STANAG 4671)
* **Single Engine Safety:** If the battery pack fails, the UAV can maintain cruise flight using the turboshaft engine alone — the sized engine's continuous rating exceeds the cruise power requirement with margin. Cruise duration in this mode is bounded by remaining fuel, consistent with the simulated mission endurance (~21.03 h nominal).
* **Silent/Emergency Loiter:** If the turboshaft engine fails, the electric motor can run on battery power to sustain flight for **~34 minutes**, enabling emergency return-to-base (RTB).
* **Reserve Energy:** Sized to enforce a 30-minute STANAG 4671 battery-only loiter reserve requirement. Derived from an average loiter draw of 27.24 kW over a 85% usable SoC window (95% down to 10% safety floor), this maps to a 16.69 kWh capacity floor. This design constraint is verified in simulation, ensuring zero power deficits and a minimum SoC trace of 0.255, well above the 0.10 airworthiness limit.

---

## 🔬 Reproducing the Canonical Result

To reproduce the canonical sizing optimization and sensitivity analysis, run:

### 1. Run 2-gene optimization baseline
```powershell
uv run backend/optimizer.py
```

### 2. Run 6-gene (GA-tuned PSR) optimization with the locked seed
```powershell
uv run backend/optimizer.py --optimize-psr --seed 42
```

### 3. Run the sensitivity sweep on the canonical design
```powershell
uv run backend/sweep.py --engine 64.36 --battery 16.69
```

### 4. Run the regression suite
```powershell
uv run pytest backend/test_regression.py
```


