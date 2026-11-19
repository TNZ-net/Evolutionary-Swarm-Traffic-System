import os, sys, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CHECKPOINT_DIR, NETWORK_PROFILE, GENOME_MODE, GENE_SHARE_EVERY

_BASE_STRATEGIES = ["ga", "pso", "gwo", "hybrid_proximity", "hybrid_performance"]


def load_strategy_history(strategy):
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, strategy, "gen_*.json")))
    history = {}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        fitnesses = d["fitnesses"]
        if not isinstance(fitnesses, dict):
            return None
        for aid, fits in fitnesses.items():
            if fits:
                history.setdefault(aid, []).append((d["generation"], max(fits)))
    return history


def main():
    if GENOME_MODE != "per_agent":
        print("Gene-sharing only applies to per_agent mode.")
        return

    for base in _BASE_STRATEGIES:
        strategy = f"{base}_per_agent_{NETWORK_PROFILE}"
        history = load_strategy_history(strategy)
        if not history:
            print(f"  No per_agent checkpoint for {strategy}, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for aid, points in history.items():
            points.sort()
            gens, fits = zip(*points)
            ax.plot(gens, fits, marker="o", alpha=0.6, label=aid)

        max_gen = max(g for pts in history.values() for g, _ in pts)
        for g in range(GENE_SHARE_EVERY, max_gen + 1, GENE_SHARE_EVERY):
            ax.axvline(g, color="red", linestyle="--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness per Junction")
        ax.set_title(f"{base} - Per-Junction Fitness (dashed = approx. gene-share gens)",
                     fontweight="bold")
        ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
        plt.tight_layout()

        out = os.path.join(CHECKPOINT_DIR, f"gene_sharing_{base}_{NETWORK_PROFILE}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {base}: saved -> {out}")


if __name__ == "__main__":
    main()