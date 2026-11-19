import os, sys, argparse, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import ROUTE_FILES, CHECKPOINT_DIR, NETWORK_PROFILE, GENOME_MODE
from env import SumoSwarmEnv
from agent import make_agent, predict
from coordinator import Coordinator
from ga import load_population, compute_fitness, fairness_penalty
from utils import load_agent

def run_episode(env, agents, coordinator, fixed_time=False) -> dict:
    obs  = env.reset()
    coordinator.reset()
    done = False
    while not done:
        if fixed_time:
            actions = {aid: 0 for aid in env.agent_ids}
        else:
            actions = {aid: predict(agents[aid], obs[aid]) for aid in env.agent_ids}
        obs, rewards, done, _ = env.step(actions)
        try:
            import libsumo as traci
        except ImportError:
            import traci

        network_arrived = traci.simulation.getArrivedNumber() 
        for aid in env.agent_ids:
            lanes  = traci.trafficlight.getControlledLanes(aid)
            q      = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            passed = network_arrived / max(len(env.agent_ids), 1)
            real_wait = sum(traci.lane.getWaitingTime(l) for l in lanes) / max(q, 1)
            coordinator.record(aid, -rewards[aid], q, passed, 0.0, real_wait=real_wait)
    return coordinator.episode_summary()


def mean_wait(summary): return summary["__network__"]["mean_wait"]

def eval_strategy(strategy: str, env, coordinator, n_episodes) -> dict:
    """Load best checkpoint and return full metrics dict. Auto-detects
    'shared' vs 'per_agent' GENOME_MODE checkpoint format via the genome_mode
    tag written by train.py (defaults to 'shared' for older checkpoints)."""
    pattern = os.path.join(CHECKPOINT_DIR, strategy, "gen_*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  No checkpoint found for {strategy}, skipping.")
        return None

    with open(files[-1]) as f:
        raw = json.load(f)
    genome_mode = raw.get("genome_mode", "shared")
 
    ckpt_dir = os.path.join(CHECKPOINT_DIR, strategy)
    if genome_mode == "per_agent":
        population, fitnesses, gen = raw["population"], raw["fitnesses"], raw["generation"]
        agents = {}
        for aid in env.agent_ids:
            if aid in population and population[aid]:
                best_idx = int(np.argmax(fitnesses.get(aid, [0])))
                fresh = make_agent(population[aid][best_idx])
                agents[aid] = load_agent(
                    os.path.join(ckpt_dir, "agents", aid, "best_agent"),
                    fallback_agent=fresh,
                )
        print(f"  {strategy}: loaded gen {gen} (per_agent, {len(agents)} junction genomes)")
    else:
        population, fitnesses, gen = load_population(files[-1])
        best_chrom = population[int(np.argmax(fitnesses))]
        agents = {}
        for aid in env.agent_ids:
            fresh = make_agent(best_chrom)
            agents[aid] = load_agent(
                os.path.join(ckpt_dir, "agents", aid, "best_agent"),
                fallback_agent=fresh,
            )
        print(f"  {strategy}: loaded gen {gen}, best fitness {max(fitnesses):.2f}")
 
    metrics = {"wait": [], "throughput": [], "clearance": [], "std_wait": [],
               "fairness_penalty": [], "wait_spread": []}

    for _ in range(n_episodes):
        summary = run_episode(env, agents, coordinator)
        net     = summary["__network__"]
        metrics["wait"].append(net["mean_wait"])
        metrics["throughput"].append(net.get("total_passed", 0))
        metrics["clearance"].append(net.get("network_clearance_time", 0))
        metrics.setdefault("peak_congestion", []).append(net.get("peak_congestion", 0))
        
        metrics["std_wait"].append(net.get("std_wait", 0))
        per_agent_waits = net.get("per_agent_waits", {})
        metrics["fairness_penalty"].append(fairness_penalty(per_agent_waits))
        metrics["wait_spread"].append(
            float(np.std(list(per_agent_waits.values()))) if per_agent_waits else 0.0
        )

    return {
        "genome_mode":      genome_mode,
        "mean_wait":        float(np.mean(metrics["wait"])),
        "std_wait":         float(np.mean(metrics["std_wait"])),
        "total_throughput": float(np.mean(metrics["throughput"])),
        "vehicles_per_hr":  float(np.mean(metrics["throughput"])) / env.step_count * 3600
                            if hasattr(env, 'step_count') and env.step_count > 0 else 0,
        "clearance_time":   float(np.mean(metrics["clearance"])),
        "peak_congestion":  float(np.mean(metrics.get("peak_congestion", [0]))),
        # spillover / fairness check: did one junction improve at a neighbour's expense?
        "fairness_penalty": float(np.mean(metrics["fairness_penalty"])),  # lower = fairer
        "wait_spread":      float(np.mean(metrics["wait_spread"])),       # high = spillover
    }

def main(args):
    from utils import set_seed
    set_seed(42)

    results = {}
    
    base_strategies = ["ga", "pso", "gwo", "hybrid_proximity", "hybrid_performance"]
    suffix = f"_per_agent_{NETWORK_PROFILE}" if GENOME_MODE == "per_agent" else f"_{NETWORK_PROFILE}"
    strategies = [f"{s}{suffix}" for s in base_strategies]
    print(f"Network: {NETWORK_PROFILE} | Genome mode: {GENOME_MODE}")
    print(f"Looking for: {strategies}")
    
    for scenario, route_file in ROUTE_FILES.items():
        print(f"\nScenario: {scenario}")
        env         = SumoSwarmEnv(route_file=route_file, seed=42)
        env.reset()
        coordinator = Coordinator(env.agent_ids)
        env.coordinator = coordinator

        # fixed-time baseline
        ft_metrics = {"wait": [], "throughput": [], "clearance": [], "std_wait": [],
                      "fairness_penalty": [], "wait_spread": []}
        for _ in range(args.episodes):
            summary = run_episode(env, {}, coordinator, fixed_time=True)
            net     = summary["__network__"]
            ft_metrics["wait"].append(net["mean_wait"])
            ft_metrics["throughput"].append(net.get("total_passed", 0))
            ft_metrics["clearance"].append(net.get("network_clearance_time", 0))
            ft_metrics["std_wait"].append(net.get("std_wait", 0))
            pa = net.get("per_agent_waits", {})
            ft_metrics["fairness_penalty"].append(fairness_penalty(pa))
            ft_metrics["wait_spread"].append(float(np.std(list(pa.values()))) if pa else 0.0)

        ft_wait = float(np.mean(ft_metrics["wait"]))
        row = {"fixed_time": {
            "mean_wait":        ft_wait,
            "std_wait":         float(np.mean(ft_metrics["std_wait"])),
            "total_throughput": float(np.mean(ft_metrics["throughput"])),
            "clearance_time":   float(np.mean(ft_metrics["clearance"])),
            "vehicles_per_hr":  0,
            "fairness_penalty": float(np.mean(ft_metrics["fairness_penalty"])),
            "wait_spread":      float(np.mean(ft_metrics["wait_spread"])),
        }}

        print(f"  fixed_time: {ft_wait:.2f}s throughput={row['fixed_time']['total_throughput']:.0f}")

        for strategy in strategies:
            m = eval_strategy(strategy, env, coordinator, args.episodes)
            if m is not None:
                row[strategy] = m
                improvement = (ft_wait - m["mean_wait"]) / ft_wait * 100
                print(f"  {strategy}: wait={m['mean_wait']:.1f}s "
                      f"throughput={m['total_throughput']:.0f} "
                      f"peak_congestion={m.get('peak_congestion', 0):.0f} "
                      f"({improvement:.1f}% better)")

        results[scenario] = row
        env.close()

    out = os.path.join(CHECKPOINT_DIR, f"comparison_{GENOME_MODE}_{NETWORK_PROFILE}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # print summary table
    print("\n── Mean Wait Time (seconds) ──")
    rows = {sc: {st: results[sc][st]["mean_wait"] for st in results[sc]}
            for sc in results}
    print(pd.DataFrame(rows).T.to_string())

    print("\n── Total Throughput (vehicles) ──")
    rows = {sc: {st: results[sc][st]["total_throughput"] for st in results[sc]}
            for sc in results}
    print(pd.DataFrame(rows).T.to_string())

    print(f"\nSaved → {out}")

    print("\nGenerating full report (comparison + Pareto)...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "report.py")])
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "pareto.py")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    main(parser.parse_args())