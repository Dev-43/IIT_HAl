"""
UAVHybridEnv — Custom Gymnasium Environment for Hybrid-Electric Fixed-Wing UAV Simulation.

Physics References:
  - Drag polar: CD = CD0 + CL² / (π · AR · e)            [Oswald model]
  - Lift coefficient: CL = 2·m·g·cos(γ) / (ρ·V²·S)
  - Aero power: P_aero = 0.5·ρ·V³·S·CD                   [= Drag × TAS]
  - Climb power: P_climb = m·g·Vz                          [Vz = V·sin(γ)]\
  - Engine Altitude Lapse (P0 fix): Gagg-Ferrar approximation
      σ = ρ(h)/ρ₀;  P_max_ice(h) = P_max_ice_SL·(σ − (1−σ)/7.55)  [clamped ≥ 0]
  - Excess-power-limited climb rate (P0 fix):
      P_excess = P_avail_thrust − P_aero;  Vz_actual = max(0, P_excess / (m·g))
  - Fuel burn: dm_fuel = SFC · P_ice · dt                  [kg]
  - Battery SoC: ΔSoC = (P_batt · dt) / E_max             [kWh / kWh]
  - Power split: PSR = P_elec / P_req;  P_ice = (1-PSR)·P_req

Data Integration:
  Component specifications (engine, battery, motor, aerodynamics) are loaded from external JSON config files; mission-phase control logic, penalty curves, and standard physical thresholds are defined inline in the simulation environment.
"""

import os
import json
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UAVHybridEnv(gym.Env):
    """
    Custom Gymnasium environment for Hybrid-Electric Fixed-Wing UAV.
    Implements full mission profile: Takeoff → Climb → Cruise → Loiter → Descent → Landing.
    """
    metadata = {"render_modes": ["human"]}

    # Mission phase constants
    PHASE_TAKEOFF = "takeoff"
    PHASE_CLIMB = "climb"
    PHASE_CRUISE = "cruise"
    PHASE_LOITER = "loiter"
    PHASE_DESCENT = "descent"
    PHASE_LANDING = "landing"
    PHASE_COMPLETED = "completed"

    def __init__(
        self,
        engine_size_kw: float,
        battery_capacity_kwh: float,
        target_speed_kmh: float = 250.0,
        target_altitude: float = 5000.0,
        payload_weight: float = 200.0,
        data_dir: str = None,
        use_heuristic_policy: bool = True,
        dt: float = 10.0,
        enable_loiter: bool = True,
        initial_fuel_fraction: float = 1.0,
        sfc_scale: float = 1.0,
        battery_density_scale: float = 1.0,
        phase_psrs: dict = None,
    ):
        super().__init__()
        self.phase_psrs = phase_psrs

        # Store sizing parameters
        self.engine_size_kw = engine_size_kw
        self.battery_capacity_kwh = battery_capacity_kwh
        self.target_speed_ms = target_speed_kmh / 3.6   # Convert km/h → m/s
        self.target_altitude = target_altitude
        self.payload_weight = payload_weight
        self.use_heuristic_policy = use_heuristic_policy
        self.dt = dt
        self.enable_loiter = enable_loiter
        self.initial_fuel_fraction = max(0.1, min(1.0, initial_fuel_fraction))
        self.sfc_scale = sfc_scale
        self.battery_density_scale = battery_density_scale

        # Resolve data directory
        if data_dir is None:
            self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        else:
            self.data_dir = data_dir

        # Load all physical constants from JSON
        self._load_constants()

        # Derived aerodynamic constants
        self.aspect_ratio = (self.aero["wingspan_m"] ** 2) / self.aero["wing_area_m2"]  # AR = b²/S
        self.g = self.aero["gravity_m_s2"]

        # ---- Weight Model ----
        # Engine weight scales linearly from reference spec
        ref_power = self.engine_specs["reference_power_kw"]
        ref_weight = self.engine_specs["reference_weight_kg"]
        self.weight_engine = (self.engine_size_kw / ref_power) * ref_weight

        # Motor weight is fixed (off-the-shelf EMRAX)
        self.weight_motor = self.motor_specs["mass_kg"]

        # Battery weight from energy density
        energy_density = self.battery_specs["energy_density_wh_per_kg"] * self.battery_density_scale
        self.weight_battery = (self.battery_capacity_kwh * 1000.0) / energy_density

        # Fixed weights
        self.weight_airframe = self.aero["airframe_mass_kg"]
        self.weight_payload = self.payload_weight
        self.mtow = self.aero["max_takeoff_weight_kg"]

        # Sum of empty weight + payload (excludes fuel)
        self.weight_empty_and_payload = (
            self.weight_airframe
            + self.weight_payload
            + self.weight_engine
            + self.weight_motor
            + self.weight_battery
        )

        # Fuel capacity = remaining mass budget, optionally capped by user fraction
        raw_fuel = self.mtow - self.weight_empty_and_payload
        self.fuel_initial = raw_fuel * self.initial_fuel_fraction

        # Motor limits from EMRAX spec
        self.motor_continuous_kw = self.motor_specs["continuous_power_kw"]
        self.motor_peak_kw = self.motor_specs["peak_power_kw"]
        self.motor_efficiency = self.motor_specs["peak_efficiency_percent"] / 100.0

        # Engine limits
        self.engine_continuous_kw = min(self.engine_size_kw,
                                         self.engine_specs["max_continuous_power_kw"] * (self.engine_size_kw / ref_power))
        self.engine_peak_kw = min(self.engine_size_kw * 1.25,
                                   self.engine_specs["peak_power_kw"] * (self.engine_size_kw / ref_power))
        self.sfc_base = self.engine_specs["specific_fuel_consumption_kg_per_kwh"] * self.sfc_scale

        # Battery C-rate limits
        self.c_rate_continuous = self.battery_specs["max_continuous_discharge_c_rate"]
        self.c_rate_peak = self.battery_specs["max_peak_discharge_c_rate"]
        self.soc_min = self.battery_specs.get("min_soc_limit", 0.10)

        # ---- Gymnasium Spaces ----
        # Observation: [Altitude(m), Speed(m/s), Battery_SoC(0-1), Fuel_Remaining(kg), P_required(kW)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([12000.0, 150.0, 1.0, 500.0, 500.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: Power Split Ratio (PSR) — -1.0=100% generator (charging), 0.0=100% Engine, 1.0=100% Battery/Motor
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Flight log for telemetry
        self.flight_log: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Data Loading                                                       #
    # ------------------------------------------------------------------ #
    def _load_constants(self):
        """Load all physical constants from /data JSON files."""
        files = {
            "aero": "aerodynamics.json",
            "engine_specs": "turboshaft_specs.json",
            "battery_specs": "battery_specs.json",
            "motor_specs": "motor_specs.json",
        }
        for attr, fname in files.items():
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Data file not found: {fpath}")
            with open(fpath, "r") as f:
                setattr(self, attr, json.load(f))

    # ------------------------------------------------------------------ #
    #  Atmosphere Model (ISA Standard Atmosphere)                         #
    # ------------------------------------------------------------------ #
    def _atmosphere(self, altitude_m: float) -> float:
        """Return air density (kg/m³) at given altitude using ISA model."""
        rho_0 = self.aero["air_density_sea_level_kg_m3"]
        if altitude_m < 11000:
            # Troposphere: ρ = ρ₀ · (1 − 2.25577×10⁻⁵·h)^4.25588
            rho = rho_0 * ((1.0 - 2.25577e-5 * altitude_m) ** 4.25588)
        else:
            # Stratosphere approximation
            rho = 0.3639 * math.exp(-(altitude_m - 11000) / 6341.6)
        return max(0.05, rho)

    # ------------------------------------------------------------------ #
    #  Stall Speed                                                        #
    # ------------------------------------------------------------------ #
    def _stall_speed(self, weight: float, rho: float, cl_max: float = 1.5) -> float:
        """Compute stall speed: V_stall = sqrt(2·W·g / (ρ·S·CL_max))."""
        S = self.aero["wing_area_m2"]
        return math.sqrt((2.0 * weight * self.g) / (rho * S * cl_max))

    # ------------------------------------------------------------------ #
    #  Propeller Efficiency by Phase                                      #
    # ------------------------------------------------------------------ #
    def _prop_efficiency(self, phase: str) -> float:
        """Return propeller efficiency for current flight phase."""
        key_map = {
            self.PHASE_TAKEOFF: "propeller_efficiency_takeoff",
            self.PHASE_CLIMB: "propeller_efficiency_climb",
            self.PHASE_CRUISE: "propeller_efficiency_cruise",
            self.PHASE_LOITER: "propeller_efficiency_cruise",
            self.PHASE_DESCENT: "propeller_efficiency_descent",
            self.PHASE_LANDING: "propeller_efficiency_descent",
        }
        key = key_map.get(phase, "propeller_efficiency_cruise")
        return self.aero[key]

    # ------------------------------------------------------------------ #
    #  Power Required (Core Physics)                                      #
    # ------------------------------------------------------------------ #
    def _compute_power_required(self, weight: float, speed: float, altitude: float,
                                 climb_rate: float, phase: str) -> tuple[float, float, float]:
        """
        Compute total shaft power required using the governing equations.
        Returns: (p_req_kw, p_aero_kw, p_climb_kw)
        """
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

    # ------------------------------------------------------------------ #
    #  SFC with Partial-Load Penalty                                      #
    # ------------------------------------------------------------------ #
    def _effective_sfc(self, engine_power_kw: float) -> float:
        """
        Apply partial-load penalty to SFC.
        If engine runs below 50% of continuous rating, SFC degrades.
        """
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

    # ------------------------------------------------------------------ #
    #  Heuristic Power-Split Policy (Zhang et al.)                        #
    # ------------------------------------------------------------------ #
    def _heuristic_psr(self, phase: str, soc: float, fuel_ratio: float) -> float:
        """
        Intelligent heuristic power management inspired by Zhang et al.:
          - Takeoff/Climb: High PSR → Motor handles torque peaks (avoids oversizing engine)
          - Cruise: Low PSR → Engine at optimal SFC (best fuel economy)
          - Loiter: Near-zero PSR → Pure engine at high load fraction
          - Descent/Landing: Zero PSR → Minimal power, engine idle
        Adjusts dynamically based on battery SoC and remaining fuel.
        """
        if phase == self.PHASE_TAKEOFF:
            # Heavy motor usage for peak torque demand
            base_psr = 0.65
            # Reduce if battery is getting low
            if soc < 0.3:
                base_psr = 0.3
            return base_psr

        elif phase == self.PHASE_CLIMB:
            # Motor assists engine for sustained climb
            base_psr = 0.50
            if soc < 0.3:
                base_psr = 0.2
            elif soc < 0.5:
                base_psr = 0.35
            return base_psr

        elif phase == self.PHASE_CRUISE:
            # Engine primary — best SFC at high load
            base_psr = 0.12
            # If battery is still very full, use some electric
            if soc > 0.8:
                base_psr = 0.20
            elif soc < 0.3:
                base_psr = 0.0  # Conserve battery for emergencies
            return base_psr

        elif phase == self.PHASE_LOITER:
            # Pure engine at near-optimal load for maximum endurance
            base_psr = 0.0
            # If fuel is critically low but battery has charge, use battery
            if fuel_ratio < 0.1 and soc > 0.3:
                base_psr = 0.5
            return base_psr

        else:  # Descent and Landing
            # Minimal power needed; engine idle
            return 0.0

    # ------------------------------------------------------------------ #
    #  Battery Power Limit (C-rate Bounded)                               #
    # ------------------------------------------------------------------ #
    def _max_battery_power(self, is_peak: bool = False) -> float:
        """Max power draw from battery, limited by C-rate and motor rating."""
        c_rate = self.c_rate_peak if is_peak else self.c_rate_continuous
        p_crate_limit = self.battery_capacity_kwh * c_rate  # kW
        motor_limit = self.motor_peak_kw if is_peak else self.motor_continuous_kw
        return min(p_crate_limit, motor_limit)

    # ------------------------------------------------------------------ #
    #  Engine Power Limit (Altitude-Derated via Gagg-Ferrar)              #
    # ------------------------------------------------------------------ #
    def _max_engine_power(self, altitude_m: float, is_peak: bool = False) -> float:
        """
        Compute altitude-derated maximum available engine power in kW using the Gagg-Ferrar formula.
        Ref: sigma = rho(h) / rho_0
             P_max_ice(h) = P_max_ice_SL * (sigma - (1 - sigma) / 7.55)
        """
        rho = self._atmosphere(altitude_m)
        rho_0 = self.aero["air_density_sea_level_kg_m3"]
        sigma = rho / rho_0
        base_power = self.engine_peak_kw if is_peak else self.engine_continuous_kw
        derated_power = base_power * (sigma - (1.0 - sigma) / 7.55)
        return max(0.0, derated_power)

    # ------------------------------------------------------------------ #
    #  Telemetry Logging                                                  #
    # ------------------------------------------------------------------ #
    def _log_telemetry(self, psr: float, p_req: float, p_motor: float,
                        p_engine: float, p_delivered: float, p_deficit: float,
                        p_aero: float = 0.0, p_climb: float = 0.0, climb_rate: float = 0.0):
        self.flight_log.append({
            "time": round(self.time_elapsed, 2),
            "altitude": round(self.altitude, 1),
            "speed": round(self.speed, 2),
            "power_required": round(p_req, 3),
            "power_delivered": round(p_delivered, 3),
            "power_motor": round(p_motor, 3),
            "power_engine": round(p_engine, 3),
            "soc": round(self.soc, 5),
            "fuel": round(self.fuel_remaining, 4),
            "weight": round(self.weight_empty_and_payload + self.fuel_remaining, 2),
            "phase": self.current_phase,
            "deficit": round(p_deficit, 3),
            "u": round(psr, 4),
            "p_aero": round(p_aero, 3),
            "p_climb": round(p_climb, 3),
            "climb_rate": round(climb_rate, 3),
        })

    # ------------------------------------------------------------------ #
    #  Observation                                                        #
    # ------------------------------------------------------------------ #
    def _get_obs(self, power_required: float) -> np.ndarray:
        return np.array(
            [self.altitude, self.speed, self.soc, self.fuel_remaining, power_required],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.altitude = 0.0
        self.speed = 0.0
        self.soc = self.battery_specs.get("max_soc_limit", 0.95)
        self.fuel_remaining = max(0.0, self.fuel_initial)
        self.time_elapsed = 0.0
        self.current_phase = self.PHASE_TAKEOFF
        self.deficit_counter = 0
        self.power_deficit_flag = False
        self.climb_prevented = False
        self.flight_log = []

        # Initial telemetry log
        self._log_telemetry(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)

        return self._get_obs(0.0), {}

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #
    def step(self, action):
        dt = self.dt

        # Immediate termination if MTOW exceeded (no fuel capacity)
        if self.fuel_initial <= 0.0:
            obs = self._get_obs(0.0)
            self._log_telemetry(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return obs, -1000.0, True, False, {"reason": "MTOW exceeded — no fuel capacity remaining."}

        # ---- Determine PSR ----
        if self.use_heuristic_policy:
            fuel_ratio = self.fuel_remaining / self.fuel_initial if self.fuel_initial > 0 else 0
            psr = self._heuristic_psr(self.current_phase, self.soc, fuel_ratio)
        elif self.phase_psrs is not None:
            psr = float(np.clip(self.phase_psrs.get(self.current_phase, 0.0), -1.0, 1.0))
        else:
            psr = float(np.clip(action[0], -1.0, 1.0))

        # ---- Phase-Dependent Flight State ----
        current_weight = self.weight_empty_and_payload + self.fuel_remaining
        rho = self._atmosphere(self.altitude)
        v_stall = self._stall_speed(current_weight, rho)

        # Determine speed and climb rate based on mission phase
        is_peak_phase = False
        climb_rate_target = 0.0
        self.power_deficit_flag = False

        if self.current_phase == self.PHASE_TAKEOFF:
            self.speed = max(1.15 * v_stall, 25.0)
            climb_rate_target = 3.0
            is_peak_phase = True
        elif self.current_phase == self.PHASE_CLIMB:
            self.speed = max(1.3 * v_stall, 35.0)
            climb_rate_target = 5.0
            is_peak_phase = True
        elif self.current_phase == self.PHASE_CRUISE:
            self.speed = max(self.target_speed_ms, 1.25 * v_stall)
            climb_rate = 0.0
        elif self.current_phase == self.PHASE_LOITER:
            # Best endurance speed ≈ 76% of cruise speed
            loiter_speed = 0.76 * self.target_speed_ms
            self.speed = max(loiter_speed, 1.2 * v_stall)
            climb_rate = 0.0
        elif self.current_phase == self.PHASE_DESCENT:
            # Realistic idle descent: ~-1.5 m/s (~295 fpm) glide-idle approach
            # At -3.0 m/s the UAV plummeted from 5000m in only 28 min — unrealistic for a 1-tonne airframe
            self.speed = max(1.2 * v_stall, 30.0)
            climb_rate = -1.5
        elif self.current_phase == self.PHASE_LANDING:
            # Shallow final approach: ~-0.8 m/s (~158 fpm) stable glideslope
            self.speed = max(1.1 * v_stall, 22.0)
            climb_rate = -0.8
        else:
            self.speed = 0.0
            climb_rate = 0.0

        # Excess-power-limited climb rate (P0 fix — "Phantom Climb"):
        # Prevents altitude from increasing when available power is insufficient.
        # Uses Gagg-Ferrar altitude-derated engine power (see _max_engine_power).
        if self.current_phase in (self.PHASE_TAKEOFF, self.PHASE_CLIMB):
            # Compute aerodynamic power required for level flight at current speed/altitude
            # (Note: we pass 0.0 for climb_rate)
            _, p_aero_kw, _ = self._compute_power_required(
                current_weight, self.speed, self.altitude, 0.0, self.current_phase
            )
            
            # Available shaft power
            p_elec_avail = 0.0 if self.soc <= self.soc_min else self._max_battery_power(is_peak=is_peak_phase)
            p_engine_avail = 0.0 if self.fuel_remaining <= 0.01 else self._max_engine_power(self.altitude, is_peak=is_peak_phase)
            p_avail_shaft = p_elec_avail + p_engine_avail
            
            # Available thrust power
            eta_prop = self._prop_efficiency(self.current_phase)
            p_avail_thrust = p_avail_shaft * eta_prop
            
            # Excess thrust power (kW)
            p_excess_thrust = p_avail_thrust - p_aero_kw
            
            # Achievable climb rate (m/s)
            v_z_actual = max(0.0, p_excess_thrust * 1000.0 / (current_weight * self.g))
            
            # Actual climb rate is limited by target climb rate
            climb_rate = min(climb_rate_target, v_z_actual)
            
            # Telemetry/power deficit flag
            self.power_deficit_flag = (v_z_actual < climb_rate_target)
            self.climb_prevented = (v_z_actual <= 0.0)
        else:
            self.climb_prevented = False

        # ---- Compute Power Required ----
        p_req_kw, p_aero_kw, p_climb_kw = self._compute_power_required(
            current_weight, self.speed, self.altitude, climb_rate, self.current_phase
        )

        # ---- Apply Power Split with Physical Constraints ----
        if psr >= 0.0:
            # Desired split
            p_motor_demand = psr * p_req_kw
            p_engine_demand = (1.0 - psr) * p_req_kw

            # Motor limit (C-rate + motor rating + SoC)
            if self.soc <= self.soc_min:
                p_motor = 0.0
            else:
                max_motor = self._max_battery_power(is_peak=is_peak_phase)
                p_motor = min(p_motor_demand, max_motor)

            # Engine limit (rating + fuel availability)
            engine_max = self._max_engine_power(self.altitude, is_peak=is_peak_phase)
            if self.fuel_remaining <= 0.01:
                p_engine = 0.0
            else:
                p_engine = min(p_engine_demand, engine_max)

            # Active load-sharing: cover any deficit
            p_delivered = p_motor + p_engine
            p_deficit = p_req_kw - p_delivered

            if p_deficit > 0.01:
                # Try extra from motor
                if self.soc > self.soc_min:
                    max_motor = self._max_battery_power(is_peak=is_peak_phase)
                    extra_motor = min(p_deficit, max_motor - p_motor)
                    if extra_motor > 0:
                        p_motor += extra_motor
                        p_delivered += extra_motor
                        p_deficit -= extra_motor

                # Try extra from engine
                if self.fuel_remaining > 0.01 and p_deficit > 0.01:
                    extra_engine = min(p_deficit, engine_max - p_engine)
                    if extra_engine > 0:
                        p_engine += extra_engine
                        p_delivered += extra_engine
                        p_deficit -= extra_engine

            if p_deficit < 0.01:
                p_deficit = 0.0
        else:
            # psr < 0.0: Charging case (Motor acts as generator, engine drives both propeller and generator)
            p_motor_demand = psr * p_req_kw
            
            engine_max = self._max_engine_power(self.altitude, is_peak=is_peak_phase)
            max_charge = self._max_battery_power(is_peak=is_peak_phase)
            max_soc = self.battery_specs.get("max_soc_limit", 0.95)
            
            if self.fuel_remaining <= 0.01 or self.soc >= max_soc:
                p_motor = 0.0
                p_engine = min(p_req_kw, engine_max)
            else:
                p_engine_demand = p_req_kw - p_motor_demand
                p_engine = min(p_engine_demand, engine_max)
                
                # Motor power is negative (generator load)
                p_motor = p_req_kw - p_engine
                if p_motor < -max_charge:
                    p_motor = -max_charge
                    p_engine = p_req_kw - p_motor
                elif p_motor > 0.0 and self.soc <= self.soc_min:
                    p_motor = 0.0

            p_delivered = p_motor + p_engine
            p_deficit = max(0.0, p_req_kw - p_delivered)

        # ---- Resource Consumption ----
        dt_hours = dt / 3600.0

        # Battery drain/charge
        if p_motor > 0:
            p_batt = p_motor / self.motor_efficiency  # Electrical power from battery
            energy_drawn_kwh = p_batt * dt_hours
            soc_drain = energy_drawn_kwh / self.battery_capacity_kwh
            self.soc = max(0.0, self.soc - soc_drain)
        elif p_motor < 0:
            # Charging logic applying generator_efficiency
            gen_eff = self.battery_specs.get("generator_efficiency", 0.85)
            p_batt = p_motor * gen_eff  # p_motor is negative, electrical power is negative
            energy_drawn_kwh = p_batt * dt_hours
            soc_drain = energy_drawn_kwh / self.battery_capacity_kwh
            max_soc = self.battery_specs.get("max_soc_limit", 0.95)
            self.soc = min(max_soc, self.soc - soc_drain)

        # Fuel burn: dm = SFC(load) · P_engine · dt
        if p_engine > 0:
            sfc = self._effective_sfc(p_engine)
            fuel_burned = sfc * p_engine * dt_hours  # kg
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_burned)

        # ---- Integrate State ----
        self.altitude = max(0.0, self.altitude + climb_rate * dt)
        self.time_elapsed += dt

        # Log telemetry
        # PSR = fraction of total power delivered by the electric motor.
        # When p_req == 0 (descent/landing: gliding idle, no thrust needed),
        # PSR is physically undefined — log 0.0 to avoid showing spurious 50% in the dashboard.
        if p_req_kw > 0:
            actual_psr = min(1.0, max(0.0, p_motor / p_req_kw))
        else:
            actual_psr = 0.0
        self._log_telemetry(actual_psr, p_req_kw, p_motor, p_engine, p_delivered, p_deficit,
                            p_aero=p_aero_kw, p_climb=p_climb_kw, climb_rate=climb_rate)

        # ---- Phase Transitions ----
        fuel_ratio = self.fuel_remaining / self.fuel_initial if self.fuel_initial > 0 else 0

        if self.current_phase == self.PHASE_TAKEOFF and self.altitude >= 200.0:
            self.current_phase = self.PHASE_CLIMB

        elif self.current_phase == self.PHASE_CLIMB and self.altitude >= self.target_altitude:
            self.current_phase = self.PHASE_CRUISE

        elif self.current_phase == self.PHASE_CRUISE:
            if self.enable_loiter:
                # Transition to loiter when 60% of fuel is consumed
                if fuel_ratio < 0.40:
                    self.current_phase = self.PHASE_LOITER
            else:
                # Non-loiter: continue straight in cruise, transition directly to descent when resources low
                if (fuel_ratio < 0.08 and self.soc < 0.15) or (fuel_ratio < 0.03):
                    self.current_phase = self.PHASE_DESCENT

        elif self.current_phase == self.PHASE_LOITER:
            # Begin descent when both resources are critically low
            if (fuel_ratio < 0.08 and self.soc < 0.15) or (fuel_ratio < 0.03):
                self.current_phase = self.PHASE_DESCENT

        elif self.current_phase == self.PHASE_DESCENT and self.altitude <= 200.0:
            self.current_phase = self.PHASE_LANDING

        elif self.current_phase == self.PHASE_LANDING and self.altitude <= 5.0:
            self.current_phase = self.PHASE_COMPLETED

        # ---- Termination Checks ----
        terminated = False
        info: dict = {}

        # Stall check
        CL_actual = 0.0
        if self.speed > 1.0:
            S = self.aero["wing_area_m2"]
            CL_actual = (2.0 * current_weight * self.g) / (rho * self.speed ** 2 * S)
        if CL_actual > 1.6:
            terminated = True
            info["reason"] = f"Stall: CL={CL_actual:.2f} exceeded limit"

        if self.current_phase == self.PHASE_COMPLETED:
            terminated = True
            info["reason"] = "Landed: Mission completed successfully"

        if self.soc <= 0.01 and self.fuel_remaining <= 0.01:
            terminated = True
            info["reason"] = "Out of energy: Both battery and fuel fully depleted"

        # Power deficit accumulation
        if p_deficit > 1.0:
            self.deficit_counter += 1
            if self.deficit_counter >= 3:
                terminated = True
                info["reason"] = (
                    f"Power deficit: Required {p_req_kw:.1f} kW, delivered {p_delivered:.1f} kW"
                )
        else:
            self.deficit_counter = 0

        # Safety truncation: 30 hours
        truncated = self.time_elapsed >= 108000.0
        if truncated:
            info["reason"] = "Truncated: 30-hour safety limit reached"

        # ---- Reward ----
        reward = 1.0
        if self.current_phase == self.PHASE_CRUISE:
            reward += 2.0
        elif self.current_phase == self.PHASE_LOITER:
            reward += 2.5  # Loiter earns slightly more (endurance objective)
        elif self.current_phase == self.PHASE_COMPLETED:
            reward += 500.0  # Successful mission completion
        if p_deficit > 0:
            reward -= p_deficit * 10.0
        if CL_actual > 1.6:
            reward -= 500.0

        return self._get_obs(p_req_kw), float(reward), terminated, truncated, info
