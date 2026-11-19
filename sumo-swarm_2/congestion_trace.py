import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CHECKPOINT_DIR, ROUTE_FILES, NETWORK_PROFILE, GENOME_MODE
from env import SumoSwarmEnv
from agent import make_agent, predict
from coordinator import Coordinator
from utils import set_seed

_BASE_STRATEGIES = ["ga", "pso", "gwo", "hybrid_proximity", "hybrid_performance"]
_SUFFIX = f"_per_agent_{NETWORK_PROFILE}" if GENOME_MODE == "per_agent" else f"_{NETWORK_PROFILE}"
COLORS = {"ga": "#066C51", "pso": "#E17603", "gwo": "#080808",
          "hybrid_proximity": "#76058C", "hybrid_performance": "#0909EE"}

def load_best_agents(strategy, agent_ids):
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, strategy, "gen_*.json")))
    if not files:
        return None, 5.0
    with open(files[-1]) as f:
        raw = json.load(f)
    if raw.get("genome_mode", "shared") == "per_agent":
        population, fitnesses = raw["population"], raw["fitnesses"]
        agents, phase_holds = {}, []
        for aid in agent_ids:
            if aid in population and population[aid]:
                best_idx = int(np.argmax(fitnesses.get(aid, [0])))
                best_chrom = population[aid][best_idx]
                agents[aid] = make_agent(best_chrom)
                phase_holds.append(best_chrom.get("phase_hold", 5.0))
        return agents, (float(np.mean(phase_holds)) if phase_holds else 5.0)
    population, fitnesses = raw["population"], raw["fitnesses"]
    best_chrom = population[int(np.argmax(fitnesses))]
    return {aid: make_agent(best_chrom) for aid in agent_ids}, best_chrom.get("phase_hold", 5.0)


def trace_congestion(strategy, route_file):
    from utils import set_seed
    set_seed(42)   # reseed per-strategy, not just once globally
    try:
        import libsumo as traci
    except ImportError:
        import traci

    env = SumoSwarmEnv(route_file=route_file, seed=42)
    obs = env.reset()
    coordinator = Coordinator(env.agent_ids)
    env.coordinator = coordinator
    obs = env.reset()

    agents, phase_hold = load_best_agents(strategy, env.agent_ids)
    if agents is None:
        env.close()
        return None, None
    env._phase_hold = phase_hold

    times, counts = [], []
    done = False
    while not done:
        actions = {aid: predict(agents[aid], obs[aid]) for aid in env.agent_ids}
        obs, rewards, done, _ = env.step(actions)
        times.append(traci.simulation.getTime())
        counts.append(traci.simulation.getMinExpectedNumber())
    env.close()
    return times, counts


def main():
    set_seed(42)
    route_file = ROUTE_FILES["peak"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for base in _BASE_STRATEGIES:
        strategy = f"{base}{_SUFFIX}"
        times, counts = trace_congestion(strategy, route_file)
        if times is None:
            print(f"  No checkpoint for {strategy}, skipping.")
            continue
        print(f"  {base}: final vehicle count = {counts[-1]}, agents loaded = {len(load_best_agents(strategy, [])) if False else 'see above'}")
        ax.plot(times, counts, label=base, color=COLORS.get(base, "#888"), linewidth=1.5)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Vehicles In Network")
    ax.set_title(f"Congestion Over Time - Peak Scenario ({NETWORK_PROFILE})", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = os.path.join(CHECKPOINT_DIR, f"congestion_trace_{GENOME_MODE}_{NETWORK_PROFILE}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()