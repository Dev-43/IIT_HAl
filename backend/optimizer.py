"""
optimizer.py — DEAP Genetic Algorithm for Hybrid-Electric UAV Component Sizing.

Outer Loop: Optimizes three core design variables:
  1. engine_size_kw       — Turboshaft shaft power rating (scales weight from reference spec)
  2. battery_capacity_kwh — Battery pack energy (scales weight from energy density)
  3. motor_count          — Number of EMRAX-228-class propulsion motors (1, 2, or 4)

Battery chemistry (a discrete, judge-selectable preset — see battery_specs.json's
chemistry_presets) and generator architecture (a fixed, documented 800V DC bus choice)
are NOT GA-searched — they're categorical engineering choices, not continuous dials.
Power-sharing strategy between thermal and electric systems can optionally also be
GA-searched (optimize_power_split=True), which adds two more genes (cruise/loiter PSR)
on top of the base three; takeoff/climb/descent/landing keep the heuristic's tuned values
even in that mode.

For each candidate individual the GA:
  1. Computes total weight (airframe + payload + engine + motor(s) + battery + fuel)
  2. Checks MTOW ≤ 1000 kg constraint (fuel = remaining budget)
  3. Runs a full flight simulation through UAVHybridEnv against the judge-authored
     mission-leg profile
  4. Returns endurance (hours) as the fitness value to MAXIMIZE, penalized if the mission
     didn't land safely OR if any loiter leg's on-station requirement wasn't fully met
"""

import os
import json
import random
import numpy as np
from deap import base, creator, tools
from environment import UAVHybridEnv

# ---- DEAP Type Registration (idempotent) ---- #
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

MOTOR_COUNT_OPTIONS = [1, 2, 4]
INCOMPLETE_MISSION_PENALTY = 0.4  # reused for "didn't land" AND "loiter leg incomplete"


def _round_motor_count(raw: float) -> int:
    """Round a continuous GA gene to the nearest valid motor-count option."""
    return min(MOTOR_COUNT_OPTIONS, key=lambda opt: abs(opt - raw))


def load_bounds(data_dir: str) -> dict:
    """Load component sizing bounds dynamically from /data JSON files."""
    with open(os.path.join(data_dir, "turboshaft_specs.json"), "r") as f:
        engine_specs = json.load(f)
    with open(os.path.join(data_dir, "battery_specs.json"), "r") as f:
        battery_specs = json.load(f)

    return {
        "engine": (engine_specs.get("min_size_kw", 30.0), engine_specs.get("max_size_kw", 120.0)),
        "battery": (battery_specs.get("min_capacity_kwh", 5.0), battery_specs.get("max_capacity_kwh", 50.0)),
        "motor_count": (1.0, 4.0),
    }


def evaluate_individual(
    individual,
    mission_legs: list,
    base_elevation_m: float,
    payload_weight: float,
    data_dir: str,
    bounds: dict,
    initial_fuel_fraction: float = 1.0,
    ambient_temp_c: float = 15.0,
    turbulence_level: float = 0.0,
    silent_loiter_mode: bool = True,
    battery_chemistry: str = "Li-NCA",
    optimize_power_split: bool = False,
    disturbance: dict = None,
):
    """
    Evaluate a single GA individual by running a full flight simulation against the
    judge-authored mission profile. Returns (endurance_hours,) as a single-objective
    fitness tuple.
    """
    if optimize_power_split:
        engine_kw, battery_kwh, motor_count_raw, psr_cruise, psr_loiter = individual[:5]
        phase_psrs = {"cruise": psr_cruise, "loiter": psr_loiter}
        use_heuristic_policy = False
    else:
        engine_kw, battery_kwh, motor_count_raw = individual[:3]
        phase_psrs = None
        use_heuristic_policy = True

    # Clip to physical bounds
    engine_kw = max(bounds["engine"][0], min(engine_kw, bounds["engine"][1]))
    battery_kwh = max(bounds["battery"][0], min(battery_kwh, bounds["battery"][1]))
    motor_count = _round_motor_count(motor_count_raw)

    # Instantiate the Gymnasium environment
    try:
        env = UAVHybridEnv(
            engine_size_kw=engine_kw,
            battery_capacity_kwh=battery_kwh,
            motor_count=motor_count,
            mission_legs=mission_legs,
            base_elevation_m=base_elevation_m,
            payload_weight=payload_weight,
            data_dir=data_dir,
            use_heuristic_policy=use_heuristic_policy,
            phase_psrs=phase_psrs,
            dt=60.0,
            initial_fuel_fraction=initial_fuel_fraction,
            ambient_temp_c=ambient_temp_c,
            turbulence_level=turbulence_level,
            silent_loiter_mode=silent_loiter_mode,
            battery_chemistry=battery_chemistry,
            disturbance=disturbance,
        )
    except Exception:
        return (0.0,)

    # MTOW constraint check: fuel_initial ≤ 0 means weight budget exceeded
    if env.fuel_initial <= 0.0:
        return (0.0,)  # Massive penalty

    # Run simulation
    obs, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step([0.5])  # action ignored by heuristic/phase_psrs

    # Fitness = total flight time in hours
    endurance_hours = env.time_elapsed / 3600.0

    # Penalize: didn't land safely, OR landed but a loiter leg's on-station requirement
    # wasn't fully met. Same 0.4x constant governs both failure modes, applied once even
    # if both hold — not stacked/squared.
    reason = info.get("reason", "")
    landed = "Landed" in reason or "Mission completed" in reason
    loiter_incomplete = any(r["role"] == "loiter" and not r["completed"] for r in env.leg_results)
    if not landed or loiter_incomplete:
        endurance_hours *= INCOMPLETE_MISSION_PENALTY

    return (endurance_hours,)


def optimize_propulsion(
    mission_legs: list,
    base_elevation_m: float = 0.0,
    payload_weight: float = 200.0,
    data_dir: str = None,
    pop_size: int = 40,
    n_gen: int = 15,
    initial_fuel_fraction: float = 1.0,
    ambient_temp_c: float = 15.0,
    turbulence_level: float = 0.0,
    silent_loiter_mode: bool = True,
    battery_chemistry: str = "Li-NCA",
    optimize_power_split: bool = False,
    disturbance: dict = None,
):
    """
    Run the DEAP Genetic Algorithm to find the optimal propulsion sizing against a
    judge-authored mission profile. Returns a dict with the best engine_size_kw,
    battery_capacity_kwh, motor_count, fitness, and per-generation convergence stats
    (plus phase_psrs if optimize_power_split=True).
    """
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    bounds = load_bounds(data_dir)

    # ---- DEAP Toolbox Setup ---- #
    toolbox = base.Toolbox()

    # Attribute generators
    toolbox.register("attr_engine", random.uniform, bounds["engine"][0], bounds["engine"][1])
    toolbox.register("attr_battery", random.uniform, bounds["battery"][0], bounds["battery"][1])
    toolbox.register("attr_motor_count", random.uniform, bounds["motor_count"][0], bounds["motor_count"][1])
    toolbox.register("attr_psr", random.uniform, 0.0, 1.0)

    # Individual setup
    if optimize_power_split:
        # Individual = [engine_kw, battery_kwh, motor_count, psr_cruise, psr_loiter]
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                toolbox.attr_engine,
                toolbox.attr_battery,
                toolbox.attr_motor_count,
                toolbox.attr_psr,
                toolbox.attr_psr,
            ),
            n=1,
        )
    else:
        # Individual = [engine_kw, battery_kwh, motor_count]
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_engine, toolbox.attr_battery, toolbox.attr_motor_count),
            n=1,
        )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    toolbox.register(
        "evaluate",
        evaluate_individual,
        mission_legs=mission_legs,
        base_elevation_m=base_elevation_m,
        payload_weight=payload_weight,
        data_dir=data_dir,
        bounds=bounds,
        initial_fuel_fraction=initial_fuel_fraction,
        ambient_temp_c=ambient_temp_c,
        turbulence_level=turbulence_level,
        silent_loiter_mode=silent_loiter_mode,
        battery_chemistry=battery_chemistry,
        optimize_power_split=optimize_power_split,
        disturbance=disturbance,
    )

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    if optimize_power_split:
        # sigma: [engine, battery, motor_count, psr_cruise, psr_loiter] — small motor_count
        # sigma so mutation explores neighboring counts rather than jumping randomly.
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=[8.0, 4.0, 0.5, 0.15, 0.15], indpb=0.35)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=[8.0, 4.0, 0.5], indpb=0.35)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # ---- Bound-clipping helper ---- #
    def clip_individual(ind):
        ind[0] = max(bounds["engine"][0], min(ind[0], bounds["engine"][1]))
        ind[1] = max(bounds["battery"][0], min(ind[1], bounds["battery"][1]))
        ind[2] = max(bounds["motor_count"][0], min(ind[2], bounds["motor_count"][1]))
        if optimize_power_split:
            for i in range(3, 5):
                ind[i] = max(0.0, min(ind[i], 1.0))

    # ---- Initialize Population ---- #
    print(f"\n========================================================")
    print(f"[START] INITIATING PROPULSION OPTIMIZATION LOOP")
    print(f"========================================================")
    print(f"  Mission Legs           : {len(mission_legs)}")
    print(f"  Base Elevation         : {base_elevation_m} m")
    print(f"  Payload Weight         : {payload_weight} kg")
    print(f"  Ambient Temp / Turb.   : {ambient_temp_c}°C / {turbulence_level}")
    print(f"  Silent Loiter Mode     : {silent_loiter_mode}")
    print(f"  Battery Chemistry      : {battery_chemistry}")
    print(f"  Initial Fuel Fraction  : {initial_fuel_fraction * 100:.1f}%")
    print(f"  GA Configuration       : Pop Size = {pop_size}, Max Gen = {n_gen}")
    print(f"  Optimize Power Split   : {optimize_power_split}")
    print(f"  Engine Sizing Search   : {bounds['engine'][0]} kW to {bounds['engine'][1]} kW")
    print(f"  Battery Sizing Search  : {bounds['battery'][0]} kWh to {bounds['battery'][1]} kWh")
    print(f"  Motor Count Options    : {MOTOR_COUNT_OPTIONS}")
    print(f"--------------------------------------------------------")

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # GA Hyperparameters
    CXPB, MUTPB = 0.6, 0.35

    # Evaluate initial population
    print("[WAIT] Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    print("[SUCCESS] Initial population evaluation complete. Starting evolution.\n")

    # Per-generation convergence stats — cheap to capture (already computed each
    # generation) and directly useful for demonstrating GA convergence quality.
    fits0 = [ind.fitness.values[0] for ind in pop]
    generation_stats = [{
        "generation": 0, "max": max(fits0), "min": min(fits0), "avg": float(np.mean(fits0)),
    }]

    # ---- Generational Loop ---- #
    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                clip_individual(c1)
                clip_individual(c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                clip_individual(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        # Generational statistics
        fits = [ind.fitness.values[0] for ind in pop]
        generation_stats.append({
            "generation": gen, "max": max(fits), "min": min(fits), "avg": float(np.mean(fits)),
        })
        best_ind = hof[0]
        print(f"[GEN] Generation {gen:02d}/{n_gen:02d}:")
        print(f"   * Max Fitness (Endurance): {max(fits):.3f} hours")
        print(f"   * Min Fitness (Endurance): {min(fits):.3f} hours")
        print(f"   * Avg Fitness (Endurance): {np.mean(fits):.3f} hours")
        print(f"   * Current Best Candidate: Engine={best_ind[0]:.2f}kW, Battery={best_ind[1]:.2f}kWh, "
              f"Motors={_round_motor_count(best_ind[2])}", end="")
        if optimize_power_split:
            print(f", PSR=[cruise={best_ind[3]:.2f}, loiter={best_ind[4]:.2f}]")
        else:
            print()
        print(f"--------------------------------------------------------")

    # ---- Return Best ---- #
    best = hof[0]
    print(f"\n[SUCCESS] PROPULSION OPTIMIZATION CONVERGED!")
    print(f"[BEST] Sized Architecture:")
    print(f"   * Turboshaft Engine Size: {best[0]:.2f} kW")
    print(f"   * Battery Capacity      : {best[1]:.2f} kWh")
    print(f"   * Motor Count           : {_round_motor_count(best[2])}")
    print(f"   * Expected Endurance    : {best.fitness.values[0]:.3f} hours")
    print(f"========================================================\n")

    ret = {
        "engine_size_kw": float(best[0]),
        "battery_capacity_kwh": float(best[1]),
        "motor_count": _round_motor_count(best[2]),
        "fitness": float(best.fitness.values[0]),
        "generation_stats": generation_stats,
    }
    if optimize_power_split:
        ret["phase_psrs"] = {"cruise": float(best[3]), "loiter": float(best[4])}
    return ret


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GA optimization for UAV sizing.")
    parser.add_argument("--optimize-power-split", action="store_true", help="Also GA-search cruise/loiter PSR.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--pop-size", type=int, default=40, help="GA population size.")
    parser.add_argument("--generations", type=int, default=15, help="GA generation count.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    SAMPLE_MISSION = [
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300, "headwind_kmh": 0.0},
        {"role": "loiter", "altitude_m": 3000, "speed_kmh": 180, "duration_min": 60},
        {"role": "cruise", "altitude_m": 5000, "speed_kmh": 250, "distance_km": 300, "headwind_kmh": 0.0},
    ]

    print(f"Running GA optimizer (optimize-power-split: {args.optimize_power_split}, seed: {args.seed})...")
    result = optimize_propulsion(
        mission_legs=SAMPLE_MISSION,
        payload_weight=200.0,
        pop_size=args.pop_size,
        n_gen=args.generations,
        optimize_power_split=args.optimize_power_split,
    )
    print("Optimization complete!")
    print(f"  Engine:      {result['engine_size_kw']:.2f} kW")
    print(f"  Battery:     {result['battery_capacity_kwh']:.2f} kWh")
    print(f"  Motor Count: {result['motor_count']}")
    if "phase_psrs" in result:
        print("  PSR Policy:")
        for phase, psr in result["phase_psrs"].items():
            print(f"    {phase}: {psr:.4f}")
    print(f"  Endurance: {result['fitness']:.4f} hours")
