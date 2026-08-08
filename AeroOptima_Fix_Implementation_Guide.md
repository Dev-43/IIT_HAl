# AeroOptima Physics Fix — Agent Implementation Guide

**Target repo:** `d:\IIT_Indore\HALxOPT` (stable/first repo — confirmed running)
**Scope:** Fix P0 physics bugs only (Phantom Climb, Engine Altitude Lapse). Do NOT touch RL pipeline, frontend, or Phase 2/3 items unless explicitly instructed in a later task.
**Audience:** Autonomous coding agent. Follow steps in exact order. Do not skip validation steps. Do not refactor unrelated code. Do not rename variables, functions, or files unless explicitly instructed.

---

## 0. Ground Rules for the Agent

1. **Work in this order only:** Step 1 → Step 2 → Step 3 → Step 4 → Step 5. Do not start Step 2 until Step 1 passes its own validation. Do not start Step 3 until Step 2 passes.
2. **Before editing any file**, open and read it in full. Do not guess line numbers or function signatures from this document — verify against the actual current file content first, since line numbers may have shifted since this doc was written.
3. **Do not invent new physical constants.** All constants (g, ρ0, engine specs, battery specs) must be read from the existing `data/*.json` files or existing config already loaded in `environment.py`. If a needed constant is missing from the JSON files, stop and flag it rather than hardcoding a guessed value.
4. **Do not change units silently.** Confirm whether power in the codebase is tracked in W or kW, and mass in kg, before writing any formula. Mixing units is the single most likely failure mode in this task.
5. **After every step, run the validation script/tests specified for that step before proceeding.** If a test fails, fix the code — do not modify the test to make it pass.
6. **Make one logical change per commit/diff.** Do not combine the Phantom Climb fix and the Gagg-Ferrar fix into a single unreviewable diff, even though they are related — implement Gagg-Ferrar first (Step 1), then Phantom Climb (Step 2), since Step 2 depends on Step 1's output being correct.
7. **If at any point the fix requires changing the observation space, reward function, action space, or any RL/training file** — stop. That is out of scope for this task. Flag it and wait for confirmation.

---

## Step 1: Implement Gagg-Ferrar Engine Altitude Derating

### 1.1 Locate the code

Find where engine/ICE maximum available power is defined or used in `backend/environment.py`. Search for the variable that represents peak/rated engine shaft power (likely sourced from `turboshaft_specs.json` or similar, and previously reported near the max-power lookup / power-cap logic). Also locate the function or block computing `ρ(h)` (ISA density) — this must already exist and be correct; reuse it, do not reimplement it.

### 1.2 Formula to implement

Gagg-Ferrar approximation for piston/turboshaft altitude power lapse:

```
sigma = rho(h) / rho_0
P_max_ice(h) = P_max_ice_SL * (sigma - (1 - sigma) / 7.55)
```

Where:

- `rho(h)` = air density at current altitude (already computed elsewhere in the file — reuse it)
- `rho_0` = sea-level air density (should already be defined as a constant, likely from `aerodynamics.json`)
- `P_max_ice_SL` = rated sea-level shaft power of the engine (from `turboshaft_specs.json`)

**Clamp the result:** `P_max_ice(h) = max(P_max_ice(h), 0)` — the Gagg-Ferrar formula can go negative at extreme altitudes outside the engine's designed envelope; power cannot be negative.

**If `turboshaft_specs.json` contains its own altitude lapse curve or coefficients** (rather than expecting the generic Gagg-Ferrar constant of 7.55), use the JSON-provided curve instead of the hardcoded 7.55 constant. Check the JSON file content before finalizing this step.

### 1.3 Apply the cap

Every place in the code where `P_ice` (electric/ICE power split output) is compared against or clamped by a maximum engine power value, that maximum must now be `P_max_ice(h)` (altitude-adjusted), not the flat sea-level rating. Find all such usages — do not assume there is only one.

### 1.4 Validation for Step 1

Before moving to Step 2, verify:

- At h = 0 (sea level), `P_max_ice(0)` equals the sea-level rated power from the spec file (sanity check that the formula reduces correctly at σ=1).
- At a mid-altitude value (e.g., 3000m), `P_max_ice(h) < P_max_ice(0)` — power must strictly decrease with altitude.
- At the aircraft's documented service ceiling (if defined in `aerodynamics.json` or elsewhere), confirm `P_max_ice(h)` is still positive and non-zero (i.e., the engine can still theoretically produce power at the top of the intended operating envelope — if it can't, that's a separate design/spec issue to flag, not silently override).
- Run `backend/test_env.py` (or add a targeted unit test if none exists for this) and confirm no exceptions and no negative power values are ever produced.

Do not proceed to Step 2 until all four checks above pass.

---

## Step 2: Fix the Phantom Climb Bug

**This step depends on Step 1 being complete and correct.** The available power figure used here must already be altitude-derated.

### 2.1 Locate the code

Find the climb-phase logic in `backend/environment.py` — the block that currently applies a fixed/unconditional climb rate (e.g., `altitude += climb_rate * dt`) during the Climb phase of the mission profile.

### 2.2 Compute required vs. available power correctly

Before allowing altitude to change, compute:

```
P_req = P_aero + P_climb_target
      = D * V + m * g * V_z_target
```

Where `V_z_target` is the desired/nominal climb rate for the phase (whatever the existing target logic currently assumes).

Then compute total available propulsion power at the current timestep, using the altitude-corrected ICE cap from Step 1 plus available electric power:

```
P_available = P_elec_available + P_max_ice(h)   # from Step 1
```

(Confirm the exact composition of "available power" against the existing Power-Split section of the file — do not change how PSR/electric power availability is computed, only ensure the ICE side now uses the Step 1 output.)

### 2.3 Derive actual achievable climb rate

```
P_excess = P_available - P_aero
V_z_actual = max(0, P_excess / (m * g))
```

- If `P_excess >= 0`: the aircraft can sustain the target climb rate (or the computed `V_z_actual`, whichever is the correct existing convention — check whether the codebase expects climb rate to be capped at the target or allowed to exceed it; likely it should be capped at `min(V_z_target, V_z_actual)`).
- If `P_excess < 0`: `V_z_actual = 0`. The aircraft cannot climb this timestep. Altitude must not increase. Do not allow negative-power climbing under any circumstance.

Replace the unconditional `altitude += climb_rate * dt` with:

```
altitude += V_z_actual * dt
```

### 2.4 Handle the power-deficit case explicitly

When `P_excess < 0` occurs during the Climb phase:

- Do not crash or throw an exception.
- Do not silently allow climb.
- Set a flag or state variable (check if one already exists, e.g. `power_deficit_flag`) so this event is visible in telemetry/logs and can later be picked up by reward shaping (out of scope for this task, but the state must be observable, not swallowed).
- Confirm whether the existing phase-transition logic (Climb → Cruise) depends on reaching a target altitude within a time budget. If the aircraft cannot climb due to power deficit, this may cause the phase to never transition — that is a valid physical outcome (the sizing configuration was infeasible for this mission), not a bug. Do not add artificial workarounds to force phase transition; let it manifest as a poor/failed episode, since that is the correct signal for the outer GA sizing loop.

### 2.5 Validation for Step 2

Before considering this step complete:

- Construct a test case in `test_env.py` (or a new test file) with sufficient engine + battery sizing where `P_excess > 0` throughout climb — confirm altitude increases and matches hand-calculated `V_z_actual` within a small tolerance.
- Construct a second test case with deliberately undersized power (e.g., artificially low `P_engine` and `E_batt` inputs) where `P_excess < 0` at some point during climb — confirm altitude does NOT increase during those timesteps, and no exception is raised.
- Confirm no regression: run the full existing `test_env.py` suite and confirm all previously-passing tests still pass.

Do not proceed to Step 3 until both new test cases pass and no regressions are introduced.

---

## Step 3: Integration Check (Both Fixes Together)

1. Run a full mission simulation end-to-end (Takeoff → Climb → Cruise → Loiter → Descent → Landing) with a nominal, reasonably-sized engine/battery configuration (use whatever default/example config already exists in the repo — do not invent one).
2. Confirm the simulation completes without exceptions.
3. Confirm telemetry output (altitude, power, SoC, fuel) is monotonic and physically sane where expected — e.g., altitude should not decrease during Climb phase under adequate power, ICE max power available should visibly decrease as altitude increases in any logged/printed power-cap values.
4. Confirm the Genetic Algorithm optimizer (`backend/optimizer.py`) still runs end-to-end with these changes and does not crash — you are not changing its interface, only the environment it evaluates against, so this should be a smoke test, not a logic change.

---

## Step 4: Documentation of Changes

For each of the two fixes, add or update:

1. A short docstring/comment in `environment.py` at the modified function(s) explaining the physical model now implemented (one or two lines, referencing "Gagg-Ferrar altitude lapse" and "excess-power-limited climb rate" by name so future readers can find the relevant literature).
2. A changelog entry (create `CHANGELOG.md` at repo root if it doesn't exist, or append if it does) with:
   - Date
   - Bug fixed (name it exactly: "Phantom Climb (P0)" and "Engine Altitude Lapse (P0)")
   - Files/lines changed
   - Validation performed

Do not add extensive new documentation beyond this — keep it concise.

---

## Step 5: Final Report Back to User

After Steps 1–4 are complete and all validations pass, produce a short summary (not a new document, just a message) containing:

1. Confirmation both P0 fixes are implemented and validated.
2. The exact files and line ranges changed.
3. Test results (pass/fail) for each validation case in Steps 1.4, 2.5, and Step 3.
4. Any issues encountered that required deviation from this guide (e.g., missing constants in JSON files, ambiguous existing conventions) — flag these explicitly rather than having silently resolved them with a guess.

---

## Explicitly Out of Scope for This Task

Do not implement any of the following even if related code is nearby or seems easy to also fix while in the file:

- 9D observation space expansion (P1)
- Dense reward shaping (P1)
- Regenerative descent / wind disturbance (P2)
- Cold battery sag / thermal modeling (P3)
- RL training pipeline (`train_rl.py`, `test_rl.py`, `/api/simulate_rl`)
- Any frontend changes
- Any changes to `main.py` API contracts

If you (the agent) identify that fixing the two P0s reveals a dependency on one of the above (e.g., you discover the reward function references `climb_rate` directly and will break), stop and flag it — do not silently fix the dependency, since that expands scope beyond what was requested.
