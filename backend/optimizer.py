"""
optimizer.py — DEAP Genetic Algorithm for Hybrid-Electric UAV Component Sizing.

Outer Loop: Optimizes two design variables:
  1. engine_size_kw  — Turboshaft shaft power rating (scales weight from reference spec)
  2. battery_capacity_kwh — Battery pack energy (scales weight from energy density)

The electric motor is a FIXED off-the-shelf component (EMRAX 228, 12.3 kg).

For each candidate individual the GA:
  1. Computes total weight (airframe + payload + engine + motor + battery + fuel)
  2. Checks MTOW ≤ 1000 kg constraint (fuel = remaining budget)
  3. Runs a full flight simulation through UAVHybridEnv with heuristic power management
  4. Returns endurance (hours) as the fitness value to MAXIMIZE
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


def load_bounds(data_dir: str) -> dict:
    """Load component sizing bounds dynamically from /data JSON files."""
    with open(os.path.join(data_dir, "turboshaft_specs.json"), "r") as f:
        engine_specs = json.load(f)
    with open(os.path.join(data_dir, "battery_specs.json"), "r") as f:
        battery_specs = json.load(f)

    return {
        "engine": (engine_specs.get("min_size_kw", 30.0), engine_specs.get("max_size_kw", 120.0)),
        "battery": (battery_specs.get("min_capacity_kwh", 5.0), battery_specs.get("max_capacity_kwh", 50.0)),
    }

def evaluate_individual(
    individual,
    target_speed_kmh: float,
    target_altitude: float,
    payload_weight: float,
    data_dir: str,
    bounds: dict,
    enable_loiter: bool = True,
    initial_fuel_fraction: float = 1.0,
    optimize_psr: bool = False,
):
    """
    Evaluate a single GA individual by running a full flight simulation.
    Returns (endurance_hours,) as a single-objective fitness tuple.
    """
    if optimize_psr:
        engine_kw, battery_kwh = individual[0], individual[1]
        phase_psrs = {
            "takeoff": individual[2],
            "climb": individual[3],
            "cruise": individual[4],
            "loiter": individual[5],
        }
        use_heuristic_policy = False
    else:
        engine_kw, battery_kwh = individual[0], individual[1]
        phase_psrs = None
        use_heuristic_policy = True

    # Clip to physical bounds
    engine_kw = max(bounds["engine"][0], min(engine_kw, bounds["engine"][1]))
    battery_kwh = max(bounds["battery"][0], min(battery_kwh, bounds["battery"][1]))

    # Instantiate the Gymnasium environment
    try:
        env = UAVHybridEnv(
            engine_size_kw=engine_kw,
            battery_capacity_kwh=battery_kwh,
            target_speed_kmh=target_speed_kmh,
            target_altitude=target_altitude,
            payload_weight=payload_weight,
            data_dir=data_dir,
            use_heuristic_policy=use_heuristic_policy,
            phase_psrs=phase_psrs,
            dt=60.0,
            enable_loiter=enable_loiter,
            initial_fuel_fraction=initial_fuel_fraction,
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
        obs, reward, terminated, truncated, info = env.step([0.5])  # action ignored by heuristic

    # Fitness = total flight time in hours
    endurance_hours = env.time_elapsed / 3600.0

    # Penalize non-successful missions (didn't complete full profile to landing)
    reason = info.get("reason", "")
    if "Landed" not in reason and "Mission completed" not in reason:
        endurance_hours *= 0.4  # 60% penalty for incomplete mission

    return (endurance_hours,)

def optimize_propulsion(
    target_speed_kmh: float = 250.0,
    target_altitude: float = 5000.0,
    payload_weight: float = 200.0,
    data_dir: str = None,
    pop_size: int = 40,
    n_gen: int = 15,
    enable_loiter: bool = True,
    initial_fuel_fraction: float = 1.0,
    optimize_psr: bool = False,
):
    """
    Run the DEAP Genetic Algorithm to find the optimal propulsion sizing.
    Returns a dict with the best engine_size_kw, battery_capacity_kwh, and fitness.
    """
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    bounds = load_bounds(data_dir)

    # ---- DEAP Toolbox Setup ---- #
    toolbox = base.Toolbox()

    # Attribute generators
    toolbox.register("attr_engine", random.uniform, bounds["engine"][0], bounds["engine"][1])
    toolbox.register("attr_battery", random.uniform, bounds["battery"][0], bounds["battery"][1])
    toolbox.register("attr_psr", random.uniform, -1.0, 1.0)

    # Individual setup
    if optimize_psr:
        # Individual = [engine_kw, battery_kwh, psr_takeoff, psr_climb, psr_cruise, psr_loiter]
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                toolbox.attr_engine,
                toolbox.attr_battery,
                toolbox.attr_psr,
                toolbox.attr_psr,
                toolbox.attr_psr,
                toolbox.attr_psr,
            ),
            n=1,
        )
    else:
        # Individual = [engine_kw, battery_kwh]
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_engine, toolbox.attr_battery),
            n=1,
        )
        
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    toolbox.register(
        "evaluate",
        evaluate_individual,
        target_speed_kmh=target_speed_kmh,
        target_altitude=target_altitude,
        payload_weight=payload_weight,
        data_dir=data_dir,
        bounds=bounds,
        enable_loiter=enable_loiter,
        initial_fuel_fraction=initial_fuel_fraction,
        optimize_psr=optimize_psr,
    )

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    if optimize_psr:
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=[8.0, 4.0, 0.15, 0.15, 0.15, 0.15], indpb=0.35)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=[8.0, 4.0], indpb=0.35)
        
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ---- Bound-clipping helper ---- #
    def clip_individual(ind):
        ind[0] = max(bounds["engine"][0], min(ind[0], bounds["engine"][1]))
        ind[1] = max(bounds["battery"][0], min(ind[1], bounds["battery"][1]))
        if optimize_psr:
            for i in range(2, 6):
                ind[i] = max(-1.0, min(ind[i], 1.0))

    # ---- Initialize Population ---- #
    print(f"\n========================================================")
    print(f"[START] INITIATING PROPULSION OPTIMIZATION LOOP")
    print(f"========================================================")
    print(f"  Target Cruise Speed   : {target_speed_kmh} km/h")
    print(f"  Target Cruise Altitude: {target_altitude} m")
    print(f"  Payload Weight        : {payload_weight} kg")
    print(f"  Loiter Phase Enabled  : {enable_loiter}")
    print(f"  Initial Fuel Fraction : {initial_fuel_fraction * 100:.1f}%")
    print(f"  GA Configuration      : Pop Size = {pop_size}, Max Gen = {n_gen}")
    print(f"  Optimize PSR Policies : {optimize_psr}")
    print(f"  Engine Sizing Search  : {bounds['engine'][0]} kW to {bounds['engine'][1]} kW")
    print(f"  Battery Sizing Search : {bounds['battery'][0]} kWh to {bounds['battery'][1]} kWh")
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

        # Generational statistics calculation
        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = hof[0]
        print(f"[GEN] Generation {gen:02d}/{n_gen:02d}:")
        print(f"   * Max Fitness (Endurance): {max(fits):.3f} hours")
        print(f"   * Min Fitness (Endurance): {min(fits):.3f} hours")
        print(f"   * Avg Fitness (Endurance): {np.mean(fits):.3f} hours")
        if optimize_psr:
            print(f"   * Current Best Candidate: Engine = {best_ind[0]:.2f} kW, Battery = {best_ind[1]:.2f} kWh, PSR = [{best_ind[2]:.2f}, {best_ind[3]:.2f}, {best_ind[4]:.2f}, {best_ind[5]:.2f}]")
        else:
            print(f"   * Current Best Candidate: Engine = {best_ind[0]:.2f} kW, Battery = {best_ind[1]:.2f} kWh")
        print(f"--------------------------------------------------------")

    # ---- Return Best ---- #
    best = hof[0]
    print(f"\n[SUCCESS] PROPULSION OPTIMIZATION CONVERGED!")
    print(f"[BEST] Sized Architecture:")
    print(f"   * Turboshaft Engine Size: {best[0]:.2f} kW")
    print(f"   * Battery Capacity      : {best[1]:.2f} kWh")
    if optimize_psr:
        print(f"   * Optimized PSR Policy  : Takeoff={best[2]:.2f}, Climb={best[3]:.2f}, Cruise={best[4]:.2f}, Loiter={best[5]:.2f}")
    print(f"   * Expected Endurance    : {best.fitness.values[0]:.3f} hours")
    print(f"========================================================\n")
    
    ret = {
        "engine_size_kw": float(best[0]),
        "battery_capacity_kwh": float(best[1]),
        "fitness": float(best.fitness.values[0]),
    }
    if optimize_psr:
        ret["phase_psrs"] = {
            "takeoff": float(best[2]),
            "climb": float(best[3]),
            "cruise": float(best[4]),
            "loiter": float(best[5]),
        }
    return ret




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GA optimization for UAV sizing.")
    parser.add_argument("--optimize-psr", action="store_true", help="Optimize PSR policies (6-gene).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--pop-size", type=int, default=40, help="GA population size.")
    parser.add_argument("--generations", type=int, default=15, help="GA generation count.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Running GA optimizer (optimize-psr: {args.optimize_psr}, seed: {args.seed})...")
    result = optimize_propulsion(
        target_speed_kmh=250.0,
        target_altitude=5000.0,
        payload_weight=200.0,
        pop_size=args.pop_size,
        n_gen=args.generations,
        optimize_psr=args.optimize_psr,
    )
    print("Optimization complete!")
    print(f"  Engine:    {result['engine_size_kw']:.2f} kW")
    print(f"  Battery:   {result['battery_capacity_kwh']:.2f} kWh")
    if "phase_psrs" in result:
        print("  PSR Policy:")
        for phase, psr in result["phase_psrs"].items():
            print(f"    {phase}: {psr:.4f}")
    print(f"  Endurance: {result['fitness']:.4f} hours")
