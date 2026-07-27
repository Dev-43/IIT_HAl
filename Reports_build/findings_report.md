# HALxOPT Project Canonical Sizing Findings Report

This report summarizes the findings, changes, and verification tests completed to lock and propagate the final canonical results for the hybrid-electric UAV propulsion system sizing optimization.

---

## 📊 Sizing Optimization Results

We ran the sizing genetic algorithms using the fully corrected physics engine (with battery floor configured to **16.69 kWh**). Seeding the 6-gene optimization with **seed 42** produced reproducible results, which are compared with the 2-gene heuristic baseline below:

| Parameter / Metric | 2-Gene Heuristic Baseline | 6-Gene Optimized Design (Seed 42) | % Improvement |
| :--- | :---: | :---: | :---: |
| **Engine Sizing** | 63.47 kW | **64.36 kW** | — |
| **Battery Capacity** | 16.69 kWh | **16.69 kWh** | — |
| **Takeoff PSR** | *N/A (Heuristic)* | **0.08** (0.0806) | — |
| **Climb PSR** | *N/A (Heuristic)* | **0.04** (0.0394) | — |
| **Cruise PSR** | *N/A (Heuristic)* | **-0.65** (-0.6463) | — |
| **Loiter PSR** | *N/A (Heuristic)* | **0.04** (0.0382) | — |
| **Nominal Endurance** | 21.017 hours | **21.033 hours** | **~0.08%** |
| **Sensitivity Range** | *N/A* | **18.700 h – 23.567 h** | — |

---

## 🛠️ Code and Documentation Propagation

To ensure consistency and ease of execution across the repository, the following updates were made:

### 1. Script CLI Support
- **[optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py):** Added argument parsing support (`--optimize-psr`, `--seed`, `--pop-size`, `--generations`) to allow customized optimizations and deterministic runs with a fixed seed.
- **[sweep.py](file:///d:/IIT_Indore/HALxOPT/backend/sweep.py):** Added argument parsing support (`--engine`, `--battery`) to run sweeps against any designed parameters from the command line.

### 2. Airworthiness Claims & Documentation
- **[README.md](file:///d:/IIT_Indore/HALxOPT/README.md):**
  - Updated the STANAG airworthiness compliance section with the locked canonical numbers.
  - Rewrote the **Single Engine Safety** statement to reflect continuous rating bounds.
  - Added a **Running Locally** section with verified commands to start both the backend and frontend.
  - Added a **Reproducing the Canonical Result** section to run the optimization, sweep, and tests.
- **[test_regression.py](file:///d:/IIT_Indore/HALxOPT/backend/test_regression.py):** Added a code comment explaining that `golden_baseline.json` is a fixed regression-test fixture and must not be regenerated to match current optimum configurations.

---

## 🧪 Verification and Tests

1. **Regression Testing:**
   - Command: `uv run pytest backend/test_regression.py`
   - Result: **5/5 tests passed** successfully (0.0 deviation), proving changes did not impact the underlying physics engine core.
2. **Local Startup Verification:**
   - **Backend:** Verified that `uv run python backend/main.py` successfully initializes the Uvicorn server on `http://127.0.0.1:8000`.
   - **Frontend:** Executed `npm install` followed by `npm run dev` inside `frontend`, verifying a clean install and startup of Next.js dev server (Turbopack) on `http://localhost:3000`.
