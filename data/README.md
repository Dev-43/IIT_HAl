# AeroOptima Data Specifications — Component Configuration Database

This directory contains the dynamic JSON configuration files that feed variables into the hybrid UAV flight physics simulation environment.

---

## 📂 Database Specifications

### 1. [aerodynamics.json](file:///d:/project/HAL/data/aerodynamics.json)
Contains structural and drag polar variables for the fixed-wing UAV airframe:
* `max_takeoff_weight_kg`: Sized cap at **1000 kg**
* `payload_capacity_kg`: Tactical military payload limit of **200 kg**
* `airframe_mass_kg`: Structural airframe weight of **350 kg** (excluding engine, motor, battery, and fuel)
* `wing_area_m2` & `wingspan_m`: **14.0 m²** and **15.0 m** ($AR = 16.07$)
* `drag_coefficient_cd0` & `oswald_efficiency_factor_e`: Base parasitic drag ($C_{D0} = 0.025$) and Oswald span efficiency factor ($e = 0.85$)
* `propeller_efficiency_takeoff/climb/cruise/descent`: Propeller thrust efficiency profiles for each phase (ranging from $0.65$ to $0.85$)

### 2. [battery_specs.json](file:///d:/project/HAL/data/battery_specs.json)
Defines characteristics of the Lithium-Ion battery pack:
* `cell_format` & `chemistry`: **21700 Cylindrical** Lithium-Ion (NCA)
* `energy_density_wh_per_kg`: Sized energy density of **250 Wh/kg**
* `max_continuous_discharge_c_rate`: **3.0 C** continuous discharge
* `max_peak_discharge_c_rate`: **5.0 C** peak discharge (restricted to takeoff and climb phases)
* `min_soc_limit` & `max_soc_limit`: Safe operating State of Charge limits (**10% to 95%**)
* `min_capacity_kwh` & `max_capacity_kwh`: Sizing bounds for the GA optimizer (**5.0 to 50.0 kWh**)

### 3. [motor_specs.json](file:///d:/project/HAL/data/motor_specs.json)
Specifies parameters for the electric motor:
* `manufacturer` & `model`: **EMRAX 228 Medium Voltage** (off-the-shelf)
* `mass_kg`: Fixed component weight of **12.3 kg**
* `continuous_power_kw` & `peak_power_kw`: **55.0 kW** continuous and **109.0 kW** peak output
* `peak_efficiency_percent`: Electric conversion efficiency of **96%**

### 4. [turboshaft_specs.json](file:///d:/project/HAL/data/turboshaft_specs.json)
Specifies parameters for the hybrid turboshaft engine:
* `engine_class`: Lightweight Turboshaft / APU
* `reference_power_kw` & `reference_weight_kg`: Sizing reference point (**67.5 kW** at **35.0 kg**)
* `dry_weight_kg`: Sized dry weight (scales linearly with power rating relative to the reference point)
* `specific_fuel_consumption_kg_per_kwh`: Base SFC of **0.38 kg/kWh** at high load factors ($\ge 80\%$)
* `fuel_type` & `fuel_density_kg_per_liter`: Jet-A1 fuel with density $0.8\text{ kg/L}$
* `min_size_kw` & `max_size_kw`: Sizing bounds for the GA optimizer (**30.0 to 120.0 kW**)
