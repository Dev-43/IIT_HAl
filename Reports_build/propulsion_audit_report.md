# UAV Propulsion Optimization Codebase Audit Report
**Project:** GA Sizing Loop + Physics Simulation (IIT Indore × HAL Hackathon)

This audit report details the implementation characteristics, physics equations, algorithm loops, and hardcoded assumptions of the UAV propulsion optimization codebase.

---

## 1. GA Chromosome & Design Variables

### Individual / Chromosome Definition
The chromosome optimization uses the DEAP Genetic Algorithm library and is defined in [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L126-L133):
```python
    # Individual = [engine_kw, battery_kwh]
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_engine, toolbox.attr_battery),
        n=1,
    )
```

The individual has exactly two continuous design variables (genes):
1. **Engine Size (kW)** (`engine_kw`): Bounded by the turboshaft specs config. Generated via `random.uniform(bounds["engine"][0], bounds["engine"][1])`.
2. **Battery Capacity (kWh)** (`battery_kwh`): Bounded by the battery specs config. Generated via `random.uniform(bounds["battery"][0], bounds["battery"][1])`.

### Bounds Definition
The bounds are loaded dynamically from data spec files on lines 31-41 in [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L31-L41):
- **Engine Power Bounds**: `30.0 kW` to `120.0 kW` (loaded from `min_size_kw` and `max_size_kw` in [turboshaft_specs.json](file:///d:/IIT_Indore/HALxOPT/data/turboshaft_specs.json#L9-L10)).
- **Battery Capacity Bounds**: `5.0 kWh` to `50.0 kWh` (loaded from `min_capacity_kwh` and `max_capacity_kwh` in [battery_specs.json](file:///d:/IIT_Indore/HALxOPT/data/battery_specs.json#L10-L11)).

### Configuration Exclusivity & Hardcoded Systems
* **Engine & Battery Sizing**: Only `[engine_size_kw, battery_capacity_kwh]` are sized by the optimizer.
* **Generator Type (AC/DC)**: **Does not appear anywhere in the codebase.** There is no AC/DC logic, efficiency model, or selection variable for the generator.
* **Motor Count**: There is no motor count variable. It is implicitly fixed to a single motor. The motor weight is hardcoded in [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L88) to a single off-the-shelf component:
  ```python
  # Motor weight is fixed (off-the-shelf EMRAX)
  self.weight_motor = self.motor_specs["mass_kg"]
  ```
  This references [motor_specs.json](file:///d:/IIT_Indore/HALxOPT/data/motor_specs.json) where `"mass_kg": 12.3` (representing the EMRAX 228 Medium Voltage).
* **Power-Split Policy**: The Power-Split Ratio (PSR) policy is **not** optimized. The GA invokes the environment in heuristic mode (`use_heuristic_policy=True` in [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L72)). The action vector passed in during step evaluations is ignored (`env.step([0.5])` on line 88).

---

## 2. Power-Shaft / Propeller Efficiency Chain

### Power Required Calculation
The function computing the required shaft power is `_compute_power_required` in [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L206-L234):
```python
    def _compute_power_required(self, weight: float, speed: float, altitude: float,
                                 climb_rate: float, phase: str) -> tuple[float, float, float]:
        if speed < 1.0:
            return (0.0, 0.0, 0.0)

        rho = self._atmosphere(altitude)
        S = self.aero["wing_area_m2"]
        CD0 = self.aero["drag_coefficient_cd0"]
        e = self.aero["oswald_efficiency_factor_e"]
        AR = self.aspect_ratio

        gamma = math.asin(max(-1.0, min(1.0, climb_rate / speed)))
        CL = (2.0 * weight * self.g * math.cos(gamma)) / (rho * speed ** 2 * S)
        CL_induced_term = CL ** 2 / (math.pi * AR * e)
        CD = CD0 + CL_induced_term

        P_aero_W = 0.5 * rho * (speed ** 3) * S * CD
        P_climb_W = weight * self.g * climb_rate
        P_prop_W = P_aero_W + P_climb_W

        eta_prop = self._prop_efficiency(phase)
        P_shaft_W = max(0.0, P_prop_W / eta_prop)

        return (P_shaft_W / 1000.0, P_aero_W / 1000.0, P_climb_W / 1000.0)
```

### Propeller Efficiency Division
Propeller efficiency is applied in the simulation loop. The thrust power requirement (`P_prop_W = P_aero_W + P_climb_W`) is divided by `eta_prop` on line 231 to obtain `P_shaft_W`:
```python
P_shaft_W = max(0.0, P_prop_W / eta_prop)
```
The value of `eta_prop` is phase-dependent, retrieved via `_prop_efficiency(phase)` ([environment.py:L190-201](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L190-L201)). The raw variables are loaded from [aerodynamics.json](file:///d:/IIT_Indore/HALxOPT/data/aerodynamics.json):
- **Takeoff**: `0.65` (`propeller_efficiency_takeoff`)
- **Climb**: `0.75` (`propeller_efficiency_climb`)
- **Cruise / Loiter**: `0.85` (`propeller_efficiency_cruise`)
- **Descent / Landing**: `0.70` (`propeller_efficiency_descent`)

### Real Physics Equation Chain
1. **Air Density**:
   $$\rho = \rho_0 \cdot (1 - 2.25577 \times 10^{-5} \cdot h)^{4.25588} \quad (\text{if } h < 11000\text{ m})$$
2. **Aspect Ratio**:
   $$\text{AR} = \frac{b^2}{S} = \frac{15.0^2}{14.0} = 16.071$$
3. **Lift Coefficient**:
   $$C_L = \frac{2 \cdot m \cdot g \cdot \cos(\gamma)}{\rho \cdot V^2 \cdot S} \quad \text{where } \gamma = \arcsin\left(\frac{V_z}{V}\right)$$
4. **Drag Coefficient (Oswald Model)**:
   $$C_D = C_{D0} + \frac{C_L^2}{\pi \cdot \text{AR} \cdot e}$$
5. **Thrust Power (Aerodynamic & Climb)**:
   $$P_{\text{prop}} = \left( 0.5 \cdot \rho \cdot V^3 \cdot S \cdot C_D \right) + \left( m \cdot g \cdot V_z \right)$$
6. **Shaft Power**:
   $$P_{\text{shaft}} = \max\left(0, \frac{P_{\text{prop}}}{\eta_{\text{prop}}}\right)$$
7. **Engine Shared Demand**:
   $$P_{\text{engine\_demand}} = (1 - \text{PSR}) \cdot P_{\text{shaft}} + \text{Deficit}$$
   (restricted to $P_{\text{engine}} \le P_{\text{engine\_max}}$ and remaining fuel limits)
8. **Fuel Burn rate ($m_{\text{dot\_fuel}}$)**:
   $$m_{\text{dot\_fuel}} = \text{SFC}_{\text{effective}} \cdot P_{\text{engine}}$$

---

## 3. SFC (Fuel Conversion) Model

### SFC Function Implementation
The specific fuel consumption with a partial-load penalty is implemented in `_effective_sfc` inside [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L238-L256):
```python
    def _effective_sfc(self, engine_power_kw: float) -> float:
        if self.engine_continuous_kw <= 0:
            return self.sfc_base
        load_fraction = engine_power_kw / self.engine_continuous_kw
        if load_fraction >= 0.8:
            return self.sfc_base  # Optimal SFC
        elif load_fraction >= 0.5:
            # Mild penalty: up to 15% worse
            penalty = 1.0 + 0.15 * (0.8 - load_fraction) / 0.3
            return self.sfc_base * penalty
        else:
            # Heavy penalty: up to 40% worse at very low loads
            penalty = 1.15 + 0.25 * (0.5 - load_fraction) / 0.5
            return self.sfc_base * penalty
```

### Partial-Load Penalties
- **Load Fraction $\ge 80\%$**: No penalty. SFC multiplier = $1.0$ ($\text{SFC} = \text{SFC}_{\text{base}}$).
- **Load Fraction $50\% \le \text{LF} < 80\%$**: Mild linear penalty degrading from $1.0$ at $80\%$ load to $1.15$ ($15\%$ penalty) at $50\%$ load:
  $$\text{Multiplier} = 1.0 + 0.15 \cdot \frac{0.8 - \text{LF}}{0.3}$$
- **Load Fraction $< 50\%$**: Heavy linear penalty degrading from $1.15$ at $50\%$ load to $1.40$ ($40\%$ penalty) at zero load:
  $$\text{Multiplier} = 1.15 + 0.25 \cdot \frac{0.5 - \text{LF}}{0.5}$$

### Storage Configuration
The base SFC value (`sfc_base`) is loaded from [turboshaft_specs.json](file:///d:/IIT_Indore/HALxOPT/data/turboshaft_specs.json#L6):
- **Base SFC**: `0.38 kg/kWh` (Jet-A1).
- **SFC Penalty Interpolation Curve**: **Hardcoded** in Python (`environment.py`).

### Fuel Consumption Formula per Timestep
The fuel burned in a simulation step ($dt$) is calculated in [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L487-L490):
```python
        if p_engine > 0:
            sfc = self._effective_sfc(p_engine)
            fuel_burned = sfc * p_engine * dt_hours  # kg
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_burned)
```
- **Formula**: $\Delta m_{\text{fuel}} = \text{SFC}_{\text{effective}} \cdot P_{\text{engine}} \cdot \frac{dt}{3600}$
- **Terms**:
  - $\Delta m_{\text{fuel}}$: Fuel mass burned during step (kg).
  - $\text{SFC}_{\text{effective}}$: Effective specific fuel consumption calculated by `_effective_sfc` (kg/kWh).
  - $P_{\text{engine}}$: Actual electrical-equivalent shaft power output by engine (kW).
  - $dt$: Timestep duration in seconds (defaults to `60.0` in optimization loop).

---

## 4. Battery / Power-Sharing (PSR) Logic

### Heuristic Power-Split Ratio (PSR) Policy
The PSR value is computed dynamically per step in `_heuristic_psr` in [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L260-L307):
* **Takeoff**: Base PSR = `0.65`. If SoC < `0.3`, PSR degrades to `0.3`.
* **Climb**: Base PSR = `0.50`. If SoC < `0.3`, PSR = `0.2`. If `0.3` $\le$ SoC < `0.5`, PSR = `0.35`.
* **Cruise**: Base PSR = `0.12`. If SoC > `0.8`, PSR = `0.20`. If SoC < `0.3`, PSR = `0.0`.
* **Loiter**: Base PSR = `0.0`. If fuel ratio < `0.1` and SoC > `0.3`, PSR = `0.5`.
* **Descent / Landing**: PSR = `0.0`.

This policy is **fully hardcoded** in Python and cannot be modified by the GA sizing loop.

### Constraint Enforcement: C-rate & SoC
1. **SoC Bounds**: If battery SoC drops to or below the minimum limit (`soc_min = 0.10`), motor power is immediately cut to zero:
   ```python
   # environment.py:L438-439
   if self.soc <= self.soc_min:
       p_motor = 0.0
   ```
2. **C-rate Bounds**: Maximum battery discharge power is capped based on the configuration C-rate limit (`c_rate_continuous` or `c_rate_peak` depending on whether it is a peak phase) and the motor's power capacity:
   ```python
   # environment.py:L311-316
   def _max_battery_power(self, is_peak: bool = False) -> float:
       c_rate = self.c_rate_peak if is_peak else self.c_rate_continuous
       p_crate_limit = self.battery_capacity_kwh * c_rate  # kW
       motor_limit = self.motor_peak_kw if is_peak else self.motor_continuous_kw
       return min(p_crate_limit, motor_limit)
   ```
3. **Violations / Power Deficits**: If clipping motor power (due to SoC or C-rate limit) or engine power limits results in delivered power being less than required (`p_deficit > 1.0 kW`), the environment monitors the deficit. If a deficit persists for **3 or more consecutive steps**, the flight simulation terminates:
   ```python
   # environment.py:L559-565
   if p_deficit > 1.0:
       self.deficit_counter += 1
       if self.deficit_counter >= 3:
           terminated = True
           info["reason"] = f"Power deficit: Required {p_req_kw:.1f} kW, delivered {p_delivered:.1f} kW"
   ```
   Such early terminations result in a **60% multiplicative penalty** on the endurance fitness score in `evaluate_individual` ([optimizer.py:L95-96](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L95-L96)).

---

## 5. Fitness Function

The fitness function evaluates individuals based on simulator endurance (hours). Penalties are applied as follows:

### 1. MTOW Budget Violation (Hard Limit)
If the empty weight + payload exceeds MTOW, the initial fuel budget becomes negative. This is caught on lines 80-82 of [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L80-L82) and results in a **hard zero** fitness score (additive/subtractive terms are bypassed):
```python
    # MTOW constraint check: fuel_initial ≤ 0 means weight budget exceeded
    if env.fuel_initial <= 0.0:
        return (0.0,)  # Massive penalty
```

### 2. Successful vs. Incomplete Mission Penalty
For candidates that pass the MTOW budget check, the baseline fitness is the elapsed flight hours (`env.time_elapsed / 3600.0`). If the mission terminates prematurely (due to battery depletion, fuel depletion, climb deficit, or aerodynamic stall), a **60% multiplicative penalty** is applied:
```python
    # optimizer.py:L90-98
    # Fitness = total flight time in hours
    endurance_hours = env.time_elapsed / 3600.0

    # Penalize non-successful missions (didn't complete full profile to landing)
    reason = info.get("reason", "")
    if "Landed" not in reason and "Mission completed" not in reason:
        endurance_hours *= 0.4  # 60% penalty for incomplete mission

    return (endurance_hours,)
```

---

## 6. What's Hardcoded vs. Optimized

| Parameter | Sizing / Policy Status | Source File | Line Number |
| :--- | :--- | :--- | :--- |
| **Engine continuous power** | Optimized by GA (within 30.0 - 120.0 kW bounds) | [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L123) | L123 |
| **Battery capacity** | Optimized by GA (within 5.0 - 50.0 kWh bounds) | [optimizer.py](file:///d:/IIT_Indore/HALxOPT/backend/optimizer.py#L124) | L124 |
| **Motor count** | Hardcoded (1 motor implicitly) | [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L88) | L88 |
| **Generator type (AC/DC)** | **Not implemented / Absent from codebase** | N/A | N/A |
| **Propeller efficiency** | Config-file values (loaded from JSON per phase) | [aerodynamics.json](file:///d:/IIT_Indore/HALxOPT/data/aerodynamics.json#L9-L12) | L9-12 |
| **SFC Base rating** | Config-file value (0.38 kg/kWh) | [turboshaft_specs.json](file:///d:/IIT_Indore/HALxOPT/data/turboshaft_specs.json#L6) | L6 |
| **SFC Load penalty curve** | Hardcoded logic in Python | [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L238-L256) | L238-256 |
| **Power-Split Policy (PSR)** | Hardcoded heuristic policy | [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L260-L307) | L260-307 |
| **Airframe mass** | Config-file value (350 kg) | [aerodynamics.json](file:///d:/IIT_Indore/HALxOPT/data/aerodynamics.json#L4) | L4 |
| **Wing area** | Config-file value ($14.0\text{ m}^2$) | [aerodynamics.json](file:///d:/IIT_Indore/HALxOPT/data/aerodynamics.json#L5) | L5 |
| **Aspect ratio** | Derived calculation (16.071) | [environment.py](file:///d:/IIT_Indore/HALxOPT/backend/environment.py#L78) | L78 |

---

## 7. Uncertainty and Sensitivity Analysis

* **No Uncertainty or Sensitivity sweeps exist** in the codebase.
* There is no Monte Carlo simulation code, perturbation script, or parameter sweep testing the impact of battery energy density fluctuations or engine SFC variations.
* All runs execute using nominal values defined in `/data/*.json` and static bounds defined in `optimizer.py`.
