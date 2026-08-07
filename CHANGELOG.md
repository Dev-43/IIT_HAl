# Changelog

## 2026-08-07

### Bug Fixed: Engine Altitude Lapse (P0)

- **File changed:** `backend/environment.py`
- **New method:** `_max_engine_power(altitude_m, is_peak)` (lines ~326–337)
- **Description:** Implemented Gagg-Ferrar altitude power lapse formula:
  `σ = ρ(h)/ρ₀; P_max_ice(h) = P_max_ice_SL × (σ − (1−σ)/7.55)`, clamped to ≥ 0.
  The flat sea-level engine cap `engine_peak_kw / engine_continuous_kw` in the power-split
  section of `step()` is replaced by `_max_engine_power(self.altitude, is_peak)` in both
  the positive PSR block and the negative PSR (charging) block.
- **Validation performed:**
  - At h=0, `_max_engine_power(0)` equals sea-level rated power ✓
  - At h=3000m, power strictly < sea-level ✓
  - At service ceiling (5000m), power > 0 ✓
  - `test_env.py` Step 1.4 checks PASSED ✓

---

### Bug Fixed: Phantom Climb (P0)

- **File changed:** `backend/environment.py`
- **Description:** Replaced the unconditional `climb_rate = 5.0` (climb) / `3.0` (takeoff)
  with an excess-power-limited computation:
  `P_excess = P_avail_thrust − P_aero; Vz_actual = max(0, P_excess×1000 / (m×g))`
  `climb_rate = min(Vz_target, Vz_actual)`
  When `Vz_actual = 0`, altitude is not updated. A `power_deficit_flag` and `climb_prevented`
  state variable are set to make the deficit visible in telemetry/logs.
- **New state variables:** `self.power_deficit_flag`, `self.climb_prevented` (reset in `reset()`)
- **Validation performed:**
  - Case A (adequate sizing, 80 kW engine / 30 kWh battery): altitude increases throughout
    climb, test PASSED ✓
  - Case B (undersized, 20 kW / 2 kWh): no altitude increase when `climb_prevented=True`,
    no exception raised, test PASSED ✓
  - Full mission end-to-end (120 kW / 40 kWh): Takeoff→Climb→Cruise→Loiter→Descent→Landing
    completed in 12.17 hrs, reason="Landed: Mission completed successfully" ✓
  - GA optimizer smoke test (pop_size=4, n_gen=2): completed without crash ✓
