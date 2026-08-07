"""
UAVHybridEnv — Custom Gymnasium Environment for Hybrid-Electric Fixed-Wing UAV Simulation.

Physics References:
  - Drag polar: CD = CD0 + CL²/(π·AR·e) + CD_beta + CD_mach   [Oswald + compressibility]
  - Lift coefficient: CL = 2·m·g·cos(γ) / (ρ·V²·S)
  - Aero power: P_aero = 0.5·ρ·V³·S·CD·turbulence_factor  [= Drag × TAS, gust-perturbed]
  - Climb power: P_climb = m·g·Vz                          [Vz = V·sin(γ)]
  - Engine Altitude Lapse (P0 fix): Gagg-Ferrar approximation
      σ = ρ(h)/ρ₀;  P_max_ice(h) = P_max_ice_SL·clip(σ − (1−σ)/7.55, 0.15, 1.0)
  - Excess-power-limited climb rate (P0 fix):
      P_excess = P_avail_thrust − P_aero;  Vz_actual = max(0, P_excess / (m·g))
  - ISA density with configurable sea-level temperature (density altitude):
      T(h) = T0 − 0.0065·h;  ρ(h) = ρ₀·((273.15+T(h))/(273.15+T0))^4.25588
  - Compressibility drag (Prandtl-Glauert): CD_mach = CD0·(1/sqrt(1−M²) − 1) for 0.4<M<0.95
  - Cold-battery derating: 1%/°C capacity+power loss below +10°C, capped at 35% loss
  - Regenerative descent: P_regen = gen_efficiency · m · g · |Vz|  (Vz < 0)
  - Fuel burn: dm_fuel = SFC · P_ice · dt                  [kg]
  - Battery SoC: ΔSoC = (P_batt · dt) / E_effective         [kWh / kWh, temp-derated]
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
        mission_legs: list = None,
        payload_weight: float = 200.0,
        data_dir: str = None,
        use_heuristic_policy: bool = True,
        dt: float = 10.0,
        initial_fuel_fraction: float = 1.0,
        sfc_scale: float = 1.0,
        battery_density_scale: float = 1.0,
        phase_psrs: dict = None,
        ambient_temp_c: float = 15.0,
        turbulence_level: float = 0.0,
        silent_loiter_mode: bool = True,
        base_elevation_m: float = 0.0,
        motor_count: int = 1,
        battery_chemistry: str = "Li-NCA",
    ):
        super().__init__()
        self.phase_psrs = phase_psrs

        # Store sizing parameters
        self.engine_size_kw = engine_size_kw
        self.battery_capacity_kwh = battery_capacity_kwh
        self.motor_count = motor_count
        self.battery_chemistry = battery_chemistry
        self.payload_weight = payload_weight
        self.use_heuristic_policy = use_heuristic_policy
        self.dt = dt
        self.initial_fuel_fraction = max(0.1, min(1.0, initial_fuel_fraction))
        self.sfc_scale = sfc_scale
        self.battery_density_scale = battery_density_scale

        # ---- Environmental / realism parameters (Phase 1) ----
        self.ambient_temp_sea_level_c = ambient_temp_c
        self.turbulence_level = max(0.0, turbulence_level)
        self.silent_loiter_mode = silent_loiter_mode
        self.base_elevation_m = base_elevation_m

        # ---- Mission profile (Phase 2) ----
        if not mission_legs:
            raise ValueError("mission_legs must be a non-empty list of leg dicts.")
        self.mission_legs = self._normalize_legs(mission_legs)

        # Resolve data directory
        if data_dir is None:
            self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        else:
            self.data_dir = data_dir

        # Load all physical constants from JSON
        self._load_constants()

        # Battery chemistry preset (not GA-searched — a discrete, judge-selectable choice;
        # merges the preset's fields over the base battery_specs dict, e.g. energy density
        # and C-rate limits, leaving soc_min/generator_efficiency/capacity bounds untouched).
        chemistry_presets = self.battery_specs.get("chemistry_presets", {})
        if self.battery_chemistry in chemistry_presets:
            self.battery_specs = {**self.battery_specs, **chemistry_presets[self.battery_chemistry]}

        # Derived aerodynamic constants
        self.aspect_ratio = (self.aero["wingspan_m"] ** 2) / self.aero["wing_area_m2"]  # AR = b²/S
        self.g = self.aero["gravity_m_s2"]

        # ---- Weight Model ----
        # Engine weight scales linearly from reference spec
        ref_power = self.engine_specs["reference_power_kw"]
        ref_weight = self.engine_specs["reference_weight_kg"]
        self.weight_engine = (self.engine_size_kw / ref_power) * ref_weight

        # Motor weight/power scale linearly with motor_count (same EMRAX-class unit
        # repeated, not a new motor model — a genuine redundancy-vs-mass trade-off).
        self.weight_motor = self.motor_specs["mass_kg"] * self.motor_count

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

        # Motor limits from EMRAX spec, scaled by motor_count
        self.motor_continuous_kw = self.motor_specs["continuous_power_kw"] * self.motor_count
        self.motor_peak_kw = self.motor_specs["peak_power_kw"] * self.motor_count
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
    #  Mission Profile Normalization                                      #
    # ------------------------------------------------------------------ #
    def _normalize_legs(self, mission_legs: list) -> list:
        """
        Validate and normalize the judge-authored mission-leg list.
          - cruise legs: distance_km (station-keeping ground distance) + optional
            headwind_kmh (signed: positive=headwind, negative=tailwind for that leg).
          - loiter legs: duration_min (on-station requirement).
        Raises ValueError on any missing/invalid field — no silent defaults or clamping.
        """
        normalized = []
        for i, leg in enumerate(mission_legs):
            role = leg.get("role")
            if role not in ("cruise", "loiter"):
                raise ValueError(f"leg[{i}]: role must be 'cruise' or 'loiter', got {role!r}.")

            altitude_m = leg.get("altitude_m")
            speed_kmh = leg.get("speed_kmh")
            if altitude_m is None or speed_kmh is None:
                raise ValueError(f"leg[{i}]: altitude_m and speed_kmh are required.")
            if altitude_m <= self.base_elevation_m + 200.0:
                raise ValueError(
                    f"leg[{i}]: altitude_m ({altitude_m}) must be more than 200m above "
                    f"base_elevation_m ({self.base_elevation_m}) — otherwise it's "
                    f"indistinguishable from the takeoff/landing threshold."
                )

            entry = {
                "role": role,
                "altitude_m": float(altitude_m),
                "speed_ms": float(speed_kmh) / 3.6,
            }

            if role == "cruise":
                distance_km = leg.get("distance_km")
                if not distance_km or distance_km <= 0:
                    raise ValueError(f"leg[{i}]: cruise legs require a positive distance_km.")
                headwind_kmh = leg.get("headwind_kmh", 0.0)
                if headwind_kmh >= speed_kmh:
                    raise ValueError(
                        f"leg[{i}]: headwind_kmh ({headwind_kmh}) must be less than "
                        f"speed_kmh ({speed_kmh}) — groundspeed would be zero or negative."
                    )
                entry["distance_km"] = float(distance_km)
                entry["headwind_ms"] = float(headwind_kmh) / 3.6
            else:  # loiter
                duration_min = leg.get("duration_min")
                if not duration_min or duration_min <= 0:
                    raise ValueError(f"leg[{i}]: loiter legs require a positive duration_min.")
                entry["duration_min"] = float(duration_min)

            normalized.append(entry)
        return normalized

    # ------------------------------------------------------------------ #
    #  Atmosphere Model (ISA Standard Atmosphere, temperature-adjustable)  #
    # ------------------------------------------------------------------ #
    def _atmosphere(self, altitude_m: float) -> float:
        """
        Return air density (kg/m³) at given altitude using the ISA model, adjusted for a
        configurable sea-level temperature (density altitude).
          T(h) = T0 − 0.0065·h  [°C]
          ρ(h) = ρ₀ · ((273.15+T(h)) / (273.15+T0))^4.25588
        At the standard T0=15°C this is algebraically identical to the fixed-temperature
        formula it replaces (0.0065/288.15 = 2.25577e-5), so behavior is unchanged unless
        ambient_temp_c is actually set away from 15.
        """
        rho_0 = self.aero["air_density_sea_level_kg_m3"]
        t_sl_k = 273.15 + self.ambient_temp_sea_level_c
        if altitude_m < 11000:
            t_alt_k = max(150.0, t_sl_k - 0.0065 * altitude_m)
            rho = rho_0 * ((t_alt_k / t_sl_k) ** 4.25588)
        else:
            t_alt_k = max(150.0, t_sl_k - 0.0065 * 11000.0)
            rho_11k = rho_0 * ((t_alt_k / t_sl_k) ** 4.25588)
            rho = rho_11k * math.exp(-(altitude_m - 11000) / 6341.6)
        return max(0.05, rho)

    def _isa_temperature(self, altitude_m: float) -> float:
        """Ambient temperature (°C) at altitude via the standard lapse rate."""
        if altitude_m < 11000.0:
            return self.ambient_temp_sea_level_c - 0.0065 * altitude_m
        return self.ambient_temp_sea_level_c - 0.0065 * 11000.0

    def _speed_of_sound(self, temp_c: float) -> float:
        """a = sqrt(gamma * R * T_kelvin) = sqrt(1.4 * 287.05 * (273.15 + temp_c))."""
        t_kelvin = max(100.0, 273.15 + temp_c)
        return math.sqrt(1.4 * 287.05 * t_kelvin)

    def _mach_number(self, speed_ms: float, temp_c: float) -> float:
        a = self._speed_of_sound(temp_c)
        return speed_ms / a if a > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Stall Speed                                                        #
    # ------------------------------------------------------------------ #
    def _stall_speed(self, weight: float, rho: float, cl_max: float = 1.5) -> float:
        """Compute stall speed: V_stall = sqrt(2·W·g / (ρ·S·CL_max))."""
        S = self.aero["wing_area_m2"]
        return math.sqrt((2.0 * weight * self.g) / (rho * S * cl_max))

    # ------------------------------------------------------------------ #
    #  Propeller Efficiency by Phase (dynamic advance-ratio model)        #
    # ------------------------------------------------------------------ #
    def _static_prop_efficiency(self, phase: str) -> float:
        """Fixed per-phase efficiency lookup (fallback for near-zero speed)."""
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

    def _advance_ratio_propeller_efficiency(
        self, speed_ms: float, prop_diameter_m: float = 1.8,
        rpm: float = 2200.0, eta_max: float = 0.86, j_opt: float = 0.85,
    ) -> float:
        """
        Propeller efficiency via Advance Ratio J = V / (n·D), a parabolic efficiency curve
        peaking at eta_max near J_opt. Replaces the fixed per-phase lookup with a model that
        actually depends on airspeed.
        """
        n_rev_per_sec = rpm / 60.0
        J = speed_ms / (n_rev_per_sec * prop_diameter_m)
        eta = 4.0 * eta_max * (J / j_opt) * (1.0 - (J / (2.0 * j_opt)))
        return max(0.50, min(eta_max, eta))

    def _prop_efficiency(self, phase: str, speed_ms: float = 0.0) -> float:
        """Dynamic advance-ratio efficiency when moving, else the static per-phase lookup."""
        if speed_ms >= 1.0:
            prop_d = self.aero.get("propeller_diameter_m", 1.8)
            return self._advance_ratio_propeller_efficiency(speed_ms, prop_diameter_m=prop_d)
        return self._static_prop_efficiency(phase)

    # ------------------------------------------------------------------ #
    #  Power Required (Core Physics)                                      #
    # ------------------------------------------------------------------ #
    def _compute_power_required(self, weight: float, speed: float, altitude: float,
                                 climb_rate: float, phase: str,
                                 temp_c: float = None,
                                 turbulence_factor: float = 1.0) -> tuple[float, float, float]:
        """
        Compute total shaft power required using the governing equations.
        Returns: (p_req_kw, p_aero_kw, p_climb_kw)
        """
        if speed < 1.0:
            return (0.0, 0.0, 0.0)

        rho = self._atmosphere(altitude)
        if temp_c is None:
            temp_c = self._isa_temperature(altitude)
        S = self.aero["wing_area_m2"]
        CD0 = self.aero["drag_coefficient_cd0"]
        e = self.aero["oswald_efficiency_factor_e"]
        AR = self.aspect_ratio

        gamma = math.asin(max(-1.0, min(1.0, climb_rate / speed)))
        CL = (2.0 * weight * self.g * math.cos(gamma)) / (rho * speed ** 2 * S)
        CL_induced_term = CL ** 2 / (math.pi * AR * e)

        # Compressibility drag (Prandtl-Glauert) — zero below M=0.4, engages on fast legs
        M = self._mach_number(speed, temp_c)
        if 0.4 < M < 0.95:
            CD_mach = CD0 * ((1.0 / math.sqrt(1.0 - M ** 2)) - 1.0)
        else:
            CD_mach = 0.0

        CD = CD0 + CL_induced_term + CD_mach

        P_aero_W = 0.5 * rho * (speed ** 3) * S * CD * turbulence_factor
        P_climb_W = weight * self.g * climb_rate
        P_prop_W = P_aero_W + P_climb_W

        eta_prop = self._prop_efficiency(phase, speed_ms=speed)
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
        Heuristic power management strategy, re-tuned to pair with the Silent Loiter mode
        and the climb battery-preservation guard (`step()` caps climb PSR at 0.05
        regardless of what this method returns, and applies the loiter silent-mode
        override) — engine-primary in climb/cruise to conserve battery for loiter.

        Loiter dual-mode (MIL-OPS):
          Mode A — Silent Loiter (ICE OFF, 100% electric): PSR=1.0. Acoustic signature
            <45 dB. Used when `self.silent_loiter_mode` and SoC permits.
          Mode B — Endurance Loiter (ICE ON, 100% engine): PSR=0.0. Used once battery is
            too depleted for stealth operation.
        """
        if phase == self.PHASE_TAKEOFF:
            if soc < 0.3:
                return 0.20
            return 0.45

        elif phase == self.PHASE_CLIMB:
            # Engine is primary climb powerplant (also hard-capped at 0.05 in step())
            if soc < 0.4:
                return 0.0
            elif soc < 0.6:
                return 0.10
            return 0.15

        elif phase == self.PHASE_CRUISE:
            if soc > 0.85:
                return 0.10
            elif soc < 0.3:
                return 0.0
            return 0.05

        elif phase == self.PHASE_LOITER:
            if self.silent_loiter_mode and soc > 0.15:
                # Mode A: Silent Loiter — 100% electric, ICE off
                return 1.0
            elif fuel_ratio < 0.15 and soc > 0.12:
                # Low fuel: moderate battery assist to conserve remaining Jet-A1
                return 0.60
            else:
                # Mode B: Endurance Loiter — 100% engine, ICE on
                return 0.0

        else:  # Descent and Landing
            return 0.0

    # ------------------------------------------------------------------ #
    #  Battery Power Limit (C-rate Bounded, thermally derated)            #
    # ------------------------------------------------------------------ #
    def _max_battery_power(self, is_peak: bool = False, temp_penalty: float = 1.0) -> float:
        """Max power draw from battery, limited by C-rate, motor rating, and cold-soak derating."""
        c_rate = (self.c_rate_peak if is_peak else self.c_rate_continuous) * temp_penalty
        p_crate_limit = self.battery_capacity_kwh * temp_penalty * c_rate  # kW
        motor_limit = self.motor_peak_kw if is_peak else self.motor_continuous_kw
        return min(p_crate_limit, motor_limit)

    def _battery_temperature_penalty(self, ambient_temp_c: float) -> float:
        """
        Cold-soak battery capacity/power derating: sub-zero temperatures increase internal
        cell resistance (R_int), causing voltage sag. 1% capacity loss per degree below
        +10°C, capped at max 35% loss (0.65 min factor). No penalty at/above +10°C.
        """
        if ambient_temp_c >= 10.0:
            return 1.0
        penalty = 1.0 - (10.0 - ambient_temp_c) * 0.01
        return max(0.65, min(1.0, penalty))

    # ------------------------------------------------------------------ #
    #  Engine Power Limit (Altitude-Derated via Gagg-Ferrar)              #
    # ------------------------------------------------------------------ #
    def _max_engine_power(self, altitude_m: float, is_peak: bool = False) -> float:
        """
        Compute altitude-derated maximum available engine power in kW using the Gagg-Ferrar formula.
        Ref: sigma = rho(h) / rho_0
             P_max_ice(h) = P_max_ice_SL * clip(sigma - (1 - sigma) / 7.55, 0.15, 1.0)
        Floor of 0.15 (rather than 0) — engine power never fully dies from pure altitude
        derating alone; only binds at extreme altitude/derate combinations.
        """
        rho = self._atmosphere(altitude_m)
        rho_0 = self.aero["air_density_sea_level_kg_m3"]
        sigma = rho / rho_0
        base_power = self.engine_peak_kw if is_peak else self.engine_continuous_kw
        derate_factor = max(0.15, min(1.0, sigma - (1.0 - sigma) / 7.55))
        return base_power * derate_factor

    # ------------------------------------------------------------------ #
    #  Regenerative Descent Energy Recovery                               #
    # ------------------------------------------------------------------ #
    def _compute_regenerative_power(self, weight: float, climb_rate: float,
                                     gen_efficiency: float = 0.85) -> float:
        """
        Electrical energy recovery (kW) generated during glide descent.
        P_regen = gen_efficiency · m · g · |v_descent|   (only when climb_rate < 0)
        """
        if climb_rate >= 0:
            return 0.0
        descent_speed_ms = abs(climb_rate)
        p_pot_w = weight * self.g * descent_speed_ms
        return (p_pot_w * gen_efficiency) / 1000.0

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

        self.altitude = self.base_elevation_m
        self.speed = 0.0
        self.soc = self.battery_specs.get("max_soc_limit", 0.95)
        self.fuel_remaining = max(0.0, self.fuel_initial)
        self.time_elapsed = 0.0
        self.current_phase = self.PHASE_TAKEOFF
        self.deficit_counter = 0
        self.power_deficit_flag = False
        self.climb_prevented = False
        self.flight_log = []
        self._was_silent_loiter_active = False  # tracks Mode A->B restart transitions

        # ---- Leg-sequencer state (Phase 2) ----
        self.leg_index = 0
        self.transitioning = True          # climbing out of takeoff toward leg[0]'s altitude
        self.leg_elapsed = 0.0             # seconds on-station in the current loiter leg/extend
        self.leg_distance_flown_km = 0.0   # km on-station in the current cruise leg
        self.leg_results = []
        self.bonus_reserve_loiter_min = 0.0
        self.rtb_triggered = False         # set once resource-critical forces final RTB
        self.reserve_fuel_equivalent_min = None
        self.reserve_battery_equivalent_min = None
        self.engine_out_survivable = None
        self._min_soc_seen = self.soc

        # Initial telemetry log
        self._log_telemetry(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)

        return self._get_obs(0.0), {}

    # ------------------------------------------------------------------ #
    #  Leg-sequencer helpers                                              #
    # ------------------------------------------------------------------ #
    @property
    def on_station_min_required(self) -> float:
        """Sum of required duration across named loiter legs — derived live from
        leg_results so it's correct whether the episode lands normally or terminates
        early (power deficit, stall, etc.) partway through a later leg."""
        return sum(r["required"] for r in self.leg_results if r["role"] == "loiter")

    @property
    def on_station_min_achieved(self) -> float:
        """Sum of achieved (possibly partial) duration across named loiter legs only —
        never includes bonus_reserve_loiter_min, so a shortfall here can't be papered
        over by unrelated extend-phase time."""
        return sum(r["achieved"] for r in self.leg_results if r["role"] == "loiter")

    def _extend_leg(self) -> dict:
        """Synthetic reserve loiter after the last explicit leg, at its altitude/speed."""
        last = self.mission_legs[-1]
        return {"role": "loiter", "altitude_m": last["altitude_m"], "speed_ms": last["speed_ms"]}

    def _current_leg(self) -> dict:
        """The leg currently being transitioned to or held on-station (never during RTB)."""
        if self.leg_index < len(self.mission_legs):
            return self.mission_legs[self.leg_index]
        return self._extend_leg()

    def _start_transition_to(self, target_altitude_m: float, tolerance: float = 5.0) -> bool:
        """
        Begin a climb/descent toward target_altitude_m. If a transition is actually needed,
        sets self.transitioning=True and self.current_phase to 'climb' or 'descent'.
        Returns True if already within tolerance (transition skipped) — the caller is then
        responsible for setting current_phase to the right on-station role/LANDING itself,
        since this helper has no context for that.
        """
        if abs(self.altitude - target_altitude_m) < tolerance:
            self.transitioning = False
            self.altitude = target_altitude_m
            return True
        self.transitioning = True
        self.current_phase = self.PHASE_CLIMB if target_altitude_m > self.altitude else self.PHASE_DESCENT
        return False

    def _record_leg_result(self, completed: bool):
        """Append a {role, required, achieved, completed} entry for the leg in progress."""
        if self.leg_index >= len(self.mission_legs):
            return  # the synthetic extend/reserve loiter never gets a leg_results entry
        leg = self.mission_legs[self.leg_index]
        if leg["role"] == "cruise":
            required, achieved = leg["distance_km"], self.leg_distance_flown_km
        else:
            required, achieved = leg["duration_min"], self.leg_elapsed / 60.0
        self.leg_results.append({
            "role": leg["role"], "required": required, "achieved": achieved, "completed": completed,
        })

    def _complete_leg_and_advance(self):
        """Record the just-finished leg, advance the pointer, and begin transitioning onward
        (to the next explicit leg, or into the synthetic extend/reserve loiter)."""
        self._record_leg_result(completed=True)
        self.leg_index += 1
        self.leg_elapsed = 0.0
        self.leg_distance_flown_km = 0.0
        next_leg = self._current_leg()
        if self._start_transition_to(next_leg["altitude_m"]):
            self.current_phase = next_leg["role"]

    def _compute_reserve_metrics(self):
        """
        Post-hoc mission-summary metrics computed once the episode reaches PHASE_COMPLETED:
        fuel/battery reserve margin ("bingo fuel") and single-engine-failure survivability,
        referenced against a canonical loiter condition derived from the mission's own legs
        (not dependent on the profile actually containing a loiter leg — works for an
        all-cruise mission too).
        """
        ref_altitude = sum(l["altitude_m"] for l in self.mission_legs) / len(self.mission_legs)
        ref_speed_ms = 0.76 * (sum(l["speed_ms"] for l in self.mission_legs) / len(self.mission_legs))

        end_weight = self.weight_empty_and_payload + self.fuel_remaining
        ref_temp_c = self._isa_temperature(ref_altitude)
        ref_p_req_kw, _, _ = self._compute_power_required(
            end_weight, ref_speed_ms, ref_altitude, 0.0, self.PHASE_LOITER, temp_c=ref_temp_c,
        )

        # Fuel reserve: assume a full-engine reference burn rate
        ref_sfc = self._effective_sfc(ref_p_req_kw)
        fuel_burn_rate_kg_per_min = (ref_sfc * ref_p_req_kw) / 60.0
        self.reserve_fuel_equivalent_min = (
            self.fuel_remaining / fuel_burn_rate_kg_per_min if fuel_burn_rate_kg_per_min > 0 else float("inf")
        )

        # Battery reserve: assume a full-electric reference burn rate
        end_temp_penalty = self._battery_temperature_penalty(self._isa_temperature(self.altitude))
        end_effective_cap = self.battery_capacity_kwh * end_temp_penalty
        ref_batt_burn_rate_kwh_per_min = (ref_p_req_kw / self.motor_efficiency) / 60.0
        usable_soc_above_floor = max(0.0, self.soc - self.soc_min)
        self.reserve_battery_equivalent_min = (
            (usable_soc_above_floor * end_effective_cap) / ref_batt_burn_rate_kwh_per_min
            if ref_batt_burn_rate_kwh_per_min > 0 else float("inf")
        )

        # Engine-out survivability at the point of minimum SoC seen across the whole flight
        usable_soc_at_min = max(0.0, self._min_soc_seen - self.soc_min)
        survivable_min = (
            (usable_soc_at_min * end_effective_cap) / ref_batt_burn_rate_kwh_per_min
            if ref_batt_burn_rate_kwh_per_min > 0 else float("inf")
        )
        self.engine_out_survivable = survivable_min >= 30.0

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
        # In power-split-optimize mode (phase_psrs set), the GA only searches PSR for
        # cruise/loiter roles — takeoff/climb/descent/landing fall back to the heuristic's
        # tuned values rather than an implicit 0.0, so a partial phase_psrs dict doesn't
        # silently force 100%-engine on phases nobody asked the GA to search.
        fuel_ratio = self.fuel_remaining / self.fuel_initial if self.fuel_initial > 0 else 0
        if self.phase_psrs is not None and self.current_phase in self.phase_psrs:
            psr = float(np.clip(self.phase_psrs[self.current_phase], -1.0, 1.0))
        elif self.use_heuristic_policy or self.phase_psrs is not None:
            psr = self._heuristic_psr(self.current_phase, self.soc, fuel_ratio)
        else:
            psr = float(np.clip(action[0], -1.0, 1.0))

        # ---- Operator Override: Silent Loiter (MIL-OPS mission command) ----
        # Applies regardless of which policy branch produced `psr` above. Reduced 0.03 SoC
        # floor (below the normal soc_min=0.10) so stealth loiter can run even at low SoC.
        silent_loiter_active_now = (
            self.current_phase == self.PHASE_LOITER
            and self.silent_loiter_mode
            and self.soc > 0.03
        )
        if silent_loiter_active_now:
            psr = 1.0

        # ---- Battery Preservation Guard: cap motor draw in climb ----
        if self.current_phase == self.PHASE_CLIMB and psr > 0.05:
            psr = 0.05

        # ---- Turbulence noise on aerodynamic power (zero effect at turbulence_level=0) ----
        if self.turbulence_level > 0:
            noise = np.random.normal(0.0, 0.05 * self.turbulence_level)
            turbulence_factor = float(np.clip(1.0 + noise, 0.8, 1.2))
        else:
            turbulence_factor = 1.0

        # ---- Phase-Dependent Flight State ----
        current_weight = self.weight_empty_and_payload + self.fuel_remaining
        rho = self._atmosphere(self.altitude)
        temp_c = self._isa_temperature(self.altitude)
        temp_penalty = self._battery_temperature_penalty(temp_c)
        effective_batt_cap = self.battery_capacity_kwh * temp_penalty
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
            # Climbing toward the current leg's altitude (leg-sequencer, or the next leg
            # during a resumed climb) — same fixed climb-rate target regardless of which
            # leg is the destination.
            self.speed = max(1.3 * v_stall, 35.0)
            climb_rate_target = 5.0
            is_peak_phase = True
        elif self.current_phase == self.PHASE_DESCENT:
            # Idle glide-descent — reused for inter-leg descents, and the final RTB descent
            # back to base_elevation_m. Fixed rate; gravity provides the energy, not power
            # availability, so no excess-power limiting is needed here (unlike climb).
            self.speed = max(1.2 * v_stall, 30.0)
            climb_rate = -1.5
        elif self.current_phase == self.PHASE_LANDING:
            # Final approach — only ever entered from the RTB descent, at base_elevation_m.
            self.speed = max(1.1 * v_stall, 22.0)
            climb_rate = -0.8
        elif self.current_phase == self.PHASE_CRUISE:
            # On-station cruise leg — never reached while self.transitioning is True.
            leg = self._current_leg()
            self.speed = max(leg["speed_ms"], 1.25 * v_stall)
            climb_rate = 0.0
        elif self.current_phase == self.PHASE_LOITER:
            # On-station loiter leg, or the synthetic extend/reserve loiter after the last
            # explicit leg (best-endurance speed ≈76% of the reference leg's speed there).
            leg = self._current_leg()
            in_extend = self.leg_index >= len(self.mission_legs)
            loiter_speed = 0.76 * leg["speed_ms"] if in_extend else leg["speed_ms"]
            self.speed = max(loiter_speed, 1.2 * v_stall)
            climb_rate = 0.0
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
                current_weight, self.speed, self.altitude, 0.0, self.current_phase,
                temp_c=temp_c, turbulence_factor=turbulence_factor,
            )

            # Available shaft power (battery side derated by cold-soak temp_penalty)
            p_elec_avail = 0.0 if self.soc <= self.soc_min else self._max_battery_power(is_peak=is_peak_phase, temp_penalty=temp_penalty)
            p_engine_avail = 0.0 if self.fuel_remaining <= 0.01 else self._max_engine_power(self.altitude, is_peak=is_peak_phase)
            p_avail_shaft = p_elec_avail + p_engine_avail

            # Available thrust power
            eta_prop = self._prop_efficiency(self.current_phase, speed_ms=self.speed)
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
            current_weight, self.speed, self.altitude, climb_rate, self.current_phase,
            temp_c=temp_c, turbulence_factor=turbulence_factor,
        )

        # ---- Apply Power Split with Physical Constraints ----
        if psr >= 0.0:
            # Desired split
            p_motor_demand = psr * p_req_kw
            p_engine_demand = (1.0 - psr) * p_req_kw

            # Motor limit (C-rate + motor rating + SoC; reduced 0.03 floor during silent loiter)
            motor_soc_floor = 0.03 if silent_loiter_active_now else self.soc_min
            if self.soc <= motor_soc_floor:
                p_motor = 0.0
            else:
                max_motor = self._max_battery_power(is_peak=is_peak_phase, temp_penalty=temp_penalty)
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
                if self.soc > motor_soc_floor:
                    max_motor = self._max_battery_power(is_peak=is_peak_phase, temp_penalty=temp_penalty)
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
            max_charge = self._max_battery_power(is_peak=is_peak_phase, temp_penalty=temp_penalty)
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

        # ---- Silent-loiter restart penalty (Mode A -> Mode B, engine off -> on) ----
        # A real turboshaft doesn't relight instantly and for free: charge a small one-time
        # fuel spike and ramp this step's engine output down, instead of an instant jump.
        restarting_engine = self._was_silent_loiter_active and not silent_loiter_active_now
        if restarting_engine and p_engine > 0:
            RESTART_FUEL_SPIKE_KG = 0.05
            self.fuel_remaining = max(0.0, self.fuel_remaining - RESTART_FUEL_SPIKE_KG)
            p_engine *= 0.5  # forced ramp: half commanded power on the restart step
            p_delivered = p_motor + p_engine
            p_deficit = max(0.0, p_req_kw - p_delivered)
        self._was_silent_loiter_active = silent_loiter_active_now

        # ---- Resource Consumption ----
        dt_hours = dt / 3600.0

        # Regenerative descent recovery (zero when climb_rate >= 0)
        p_regen_kw = self._compute_regenerative_power(current_weight, climb_rate)

        # Battery drain/charge
        if p_motor > 0:
            p_batt = p_motor / self.motor_efficiency  # Electrical power from battery
            energy_drawn_kwh = (p_batt - p_regen_kw) * dt_hours
            soc_change = energy_drawn_kwh / effective_batt_cap
            self.soc = max(0.0, min(self.battery_specs.get("max_soc_limit", 0.95), self.soc - soc_change))
        elif p_motor < 0:
            # Charging logic applying generator_efficiency
            gen_eff = self.battery_specs.get("generator_efficiency", 0.85)
            p_batt = p_motor * gen_eff  # p_motor is negative, electrical power is negative
            energy_drawn_kwh = p_batt * dt_hours
            soc_drain = energy_drawn_kwh / effective_batt_cap
            max_soc = self.battery_specs.get("max_soc_limit", 0.95)
            self.soc = min(max_soc, self.soc - soc_drain)
        elif p_regen_kw > 0:
            # Idle motor (typical in descent) — regen energy credits SoC directly
            energy_recovered_kwh = p_regen_kw * dt_hours
            soc_recovered = energy_recovered_kwh / effective_batt_cap
            max_soc = self.battery_specs.get("max_soc_limit", 0.95)
            self.soc = min(max_soc, self.soc + soc_recovered)

        # Fuel burn: dm = SFC(load) · P_engine · dt
        if p_engine > 0:
            sfc = self._effective_sfc(p_engine)
            fuel_burned = sfc * p_engine * dt_hours  # kg
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_burned)

        # ---- Integrate State ----
        self.altitude = max(self.base_elevation_m, self.altitude + climb_rate * dt)
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

        # ---- Phase Transitions (leg-sequencer) ----
        fuel_ratio = self.fuel_remaining / self.fuel_initial if self.fuel_initial > 0 else 0
        resource_critical = (fuel_ratio < 0.08 and self.soc < 0.15) or (fuel_ratio < 0.03)
        self._min_soc_seen = min(self._min_soc_seen, self.soc)

        if self.current_phase == self.PHASE_TAKEOFF and self.altitude >= self.base_elevation_m + 200.0:
            # Takeoff complete -> begin transitioning toward leg[0]'s altitude. Always a
            # climb in practice: every leg is validated to sit above base_elevation_m+200.
            first_leg = self._current_leg()
            if self._start_transition_to(first_leg["altitude_m"]):
                self.current_phase = first_leg["role"]

        elif (
            self.current_phase in (self.PHASE_CLIMB, self.PHASE_CRUISE, self.PHASE_LOITER)
            and resource_critical and not self.rtb_triggered
        ):
            # Resource-critical trigger — abandon whatever's in progress and RTB. If an
            # explicit leg was on-station, record it as incomplete with its partial
            # achieved value (the failure mode that actually matters militarily); the
            # synthetic extend/reserve loiter has nothing to record.
            self.rtb_triggered = True
            if self.current_phase in (self.PHASE_CRUISE, self.PHASE_LOITER) and self.leg_index < len(self.mission_legs):
                self._record_leg_result(completed=False)
            if self._start_transition_to(self.base_elevation_m + 200.0):
                self.current_phase = self.PHASE_LANDING

        elif self.current_phase in (self.PHASE_CLIMB, self.PHASE_DESCENT) and self.transitioning:
            # Mid climb/descent — check whether the target altitude has been reached.
            target_alt = (self.base_elevation_m + 200.0) if self.rtb_triggered else self._current_leg()["altitude_m"]
            reached = (
                (self.current_phase == self.PHASE_CLIMB and self.altitude >= target_alt)
                or (self.current_phase == self.PHASE_DESCENT and self.altitude <= target_alt)
            )
            if reached:
                self.altitude = target_alt  # snap to target — avoid dt-step overshoot drift
                self.transitioning = False
                if self.rtb_triggered:
                    self.current_phase = self.PHASE_LANDING
                else:
                    self.current_phase = self._current_leg()["role"]
                    self.leg_elapsed = 0.0
                    self.leg_distance_flown_km = 0.0

        elif self.current_phase == self.PHASE_CRUISE:
            # On-station cruise leg — accumulate ground distance (headwind extends the time,
            # and therefore the energy burned, needed to cover a fixed distance).
            leg = self._current_leg()
            groundspeed_kmh = (leg["speed_ms"] * 3.6) - (leg.get("headwind_ms", 0.0) * 3.6)
            self.leg_distance_flown_km += groundspeed_kmh * dt_hours
            if self.leg_distance_flown_km >= leg["distance_km"]:
                self._complete_leg_and_advance()

        elif self.current_phase == self.PHASE_LOITER:
            # On-station loiter leg, or the synthetic extend/reserve loiter (bonus margin,
            # tracked separately so it can never paper over a genuinely incomplete leg).
            if self.leg_index >= len(self.mission_legs):
                self.bonus_reserve_loiter_min += dt / 60.0
            else:
                self.leg_elapsed += dt
                leg = self._current_leg()
                if self.leg_elapsed >= leg["duration_min"] * 60.0:
                    self._complete_leg_and_advance()

        elif self.current_phase == self.PHASE_DESCENT and self.altitude <= self.base_elevation_m + 200.0:
            self.current_phase = self.PHASE_LANDING

        elif self.current_phase == self.PHASE_LANDING and self.altitude <= self.base_elevation_m + 5.0:
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

        # Reserve/engine-out summary metrics are computed once, whenever the episode ends
        # for ANY reason — not only a successful landing — so a failure diagnosis (e.g.
        # "ran out of reserve, hence the deficit") is always available, not silently None.
        if (terminated or truncated) and self.reserve_fuel_equivalent_min is None:
            self._compute_reserve_metrics()

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
