# AeroOptima Backend — Hybrid UAV Physics & Optimization Engine

This directory contains the Python-based simulation and optimization backend for the hybrid-electric UAV.

---

## 📂 Core Files

* **[environment.py](file:///d:/project/HAL/backend/environment.py):** A custom Gymnasium-compliant environment (`UAVHybridEnv`) simulating flight mechanics.
  * *Atmosphere:* International Standard Atmosphere (ISA) model calculates dynamic air density $\rho(h)$.
  * *Aerodynamics:* Computes stall speed ($V_{stall}$), lift coefficient ($C_L$), induced drag, and total drag force.
  * *Propulsion:* Parallel hybrid system modeling motor C-rate discharge limits and turboshaft SFC (Specific Fuel Consumption) curves with low-throttle efficiency penalties.
  * *Mission Phases:* Takeoff $\to$ Climb $\to$ Cruise $\to$ Loiter $\to$ Descent $\to$ Landing $\to$ Completed.
* **[optimizer.py](file:///d:/project/HAL/backend/optimizer.py):** Implements a Genetic Algorithm (GA) using the `DEAP` library to maximize mission endurance (hours) by sizing:
  1. Turboshaft engine size ($30\text{--}120\text{ kW}$)
  2. Lithium-ion battery capacity ($5\text{--}50\text{ kWh}$)
* **[main.py](file:///d:/project/HAL/backend/main.py):** A FastAPI app exposing optimization endpoints and serving telemetry JSON to the frontend.
* **[test_env.py](file:///d:/project/HAL/backend/test_env.py):** A diagnostic CLI test script to simulate and display flight logs directly in the terminal.

---

## 📐 Physics and Governing Equations

### 1. Drag and Lift Coefficients
$$C_L = \frac{2 \cdot m \cdot g \cdot \cos(\gamma)}{\rho \cdot V^2 \cdot S}$$

$$C_D = C_{D0} + \frac{C_L^2}{\pi \cdot AR \cdot e}$$

*Where $C_{D0} = 0.025$, Oswald efficiency $e = 0.85$, and Aspect Ratio $AR = 16.07$ (15m wingspan / 14m² wing area).*

### 2. Shaft Power Required ($P_{shaft}$)
$$P_{prop} = P_{aero} + P_{climb} = \left(\frac{1}{2}\rho V^3 S C_D\right) + \left(m \cdot g \cdot V_z\right)$$

$$P_{shaft} = \frac{P_{prop}}{\eta_{prop}}$$

*Where $\eta_{prop}$ ranges from $0.65$ (takeoff) to $0.85$ (cruise).*

### 3. Engine SFC Partial-Load Penalty
To model realistic turboshaft behavior, fuel consumption scales based on throttle load. Running below $80\%$ capacity introduces an algorithmic efficiency penalty to the base SFC ($0.38\text{ kg/kWh}$):
* Load fraction $\ge 80\%$: optimal base SFC
* $50\% \le$ Load $< 80\%$: up to $+15\%$ worse SFC
* Load $< 50\%$: up to $+40\%$ worse SFC

---

## 📡 API Endpoints

### `POST /api/optimize`
Runs the DEAP optimization engine based on mission parameters, then runs a fine-grained re-simulation of the optimal design point to output telemetry.
* **Request Body:**
  ```json
  {
    "target_speed_kmh": 250.0,
    "target_altitude": 5000.0,
    "payload_weight": 200.0,
    "enable_loiter": true,
    "initial_fuel_fraction": 1.0
  }
  ```
* **Response Body:** Returns optimized sizing specs (engine kW, battery kWh, weight breakdown) and a time-series telemetry array of the entire flight profile.

### `GET /api/health`
Performs a basic API health check and returns active motor configurations.

---

## 🧪 Verification & Testing
To run a local command-line diagnostic test and inspect the mission timeline:
```powershell
.\.venv\Scripts\python test_env.py
```
