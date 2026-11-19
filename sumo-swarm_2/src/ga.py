
import numpy as np
import json, os
from copy import deepcopy
from config import (
    GENE_BOUNDS, GA_POPULATION, GA_MUTATION_RATE,
    GA_ELITE_N, HEAL_THRESHOLD, HEAL_STREAK,
)
GENE_KEYS = list(GENE_BOUNDS.keys())

# ── chromosome ────────────────────────────────────────────────────────────────
def random_chromosome() -> dict:
    chrom = {
        key: float(np.random.uniform(lo, hi))
        for key, (lo, hi) in GENE_BOUNDS.items()
    }
    # utility weights — also evolved by GA/PSO/GWO
    chrom["w_wait"]       = float(np.random.uniform(0.3, 0.6))
    chrom["w_queue"]      = float(np.random.uniform(0.1, 0.3))
    chrom["w_stops"]      = float(np.random.uniform(0.05, 0.2))
    chrom["w_throughput"] = float(np.random.uniform(0.1, 0.3))
    chrom["w_fairness"]   = float(np.random.uniform(0.05, 0.2))
    return chrom


def encode(chrom: dict) -> np.ndarray:
    all_keys = GENE_KEYS + ["w_wait", "w_queue", "w_stops", "w_throughput", "w_fairness"]
    arr = []
    for key in all_keys:
        if key in GENE_BOUNDS:
            lo, hi = GENE_BOUNDS[key]
        else:
            lo, hi = 0.0, 1.0   # utility weights already in [0,1] range
        val = chrom.get(key, (lo + hi) / 2)
        arr.append((val - lo) / (hi - lo))
    return np.array(arr, dtype=np.float32)


def decode(arr: np.ndarray) -> dict:
    all_keys = GENE_KEYS + ["w_wait", "w_queue", "w_stops", "w_throughput", "w_fairness"]
    chrom = {}
    for i, key in enumerate(all_keys):
        if key in GENE_BOUNDS:
            lo, hi = GENE_BOUNDS[key]
        else:
            lo, hi = 0.0, 1.0
        chrom[key] = float(np.clip(arr[i], 0, 1) * (hi - lo) + lo)
    return chrom


# ── multi-objective utility fitness ───────────────────────────────────────────
def compute_fitness(
    local_wait: float,
    queue_overflows: int,
    network_throughput: float,
    stops: float = 0.0,
    fairness_penalty: float = 0.0,
    wait_std=0.0,
    chromosome: dict = None,
) -> float:
    """
    Weights are taken from chromosome if provided (evolved),
    U = -w1·wait - w2·queue - w3·stops + w4·throughput - w5·fairness_penalty
    """
    w1 = chromosome.get("w_wait",       0.50) if chromosome else 0.50
    w2 = chromosome.get("w_queue",      0.20) if chromosome else 0.20
    w3 = chromosome.get("w_stops",      0.10) if chromosome else 0.10
    w4 = chromosome.get("w_throughput", 0.15) if chromosome else 0.15
    w5 = chromosome.get("w_fairness",   0.05) if chromosome else 0.05

    return (
        -w1 * local_wait
        -w2 * float(queue_overflows)
        -w3 * stops
        +w4 * network_throughput
        -w5 * fairness_penalty
        -0.05 * wait_std  
    )


def fairness_penalty(per_agent_waits: dict) -> float:
    """
    Jain's fairness index inverted — penalises unequal wait distribution.
    0 = fair, high = one approach being starved.
    """
    waits = list(per_agent_waits.values())
    if not waits or max(waits) == 0:
        return 0.0
    waits = np.array(waits)
    jain  = (np.sum(waits) ** 2) / (len(waits) * np.sum(waits ** 2) + 1e-9)
    return float(1.0 - jain)   # 0=fair, 1=completely unfair


# ── Bellman target helper ─────────────────────────────────────────────────────
def bellman_target(reward: float, next_value: float, gamma: float) -> float:
    #TD(0) Bellman target: R + γ·V(s')
    return reward + gamma * next_value

def td_lambda_returns(rewards: list, gamma: float = 0.95, lam: float = 0.9) -> list:  
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * lam * G
        returns.insert(0, G)
    return returns


# ── selection ─────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, k=3):
    idxs = np.random.choice(len(population), size=k, replace=False)
    best = max(idxs, key=lambda i: fitnesses[i])
    return deepcopy(population[best])


# ── crossover ─────────────────────────────────────────────────────────────────

def crossover(parent_a, parent_b):
    arr_a  = encode(parent_a)
    arr_b  = encode(parent_b)
    point  = np.random.randint(1, len(arr_a))
    return decode(np.concatenate([arr_a[:point], arr_b[point:]]))


# ── mutation ──────────────────────────────────────────────────────────────────

def mutate(chrom, rate=GA_MUTATION_RATE, scale=0.1):
    arr = encode(chrom)
    for i in range(len(arr)):
        if np.random.rand() < rate:
            arr[i] = np.clip(arr[i] + np.random.normal(0, scale), 0, 1)
    return decode(arr)


# ── self-healing ──────────────────────────────────────────────────────────────

class HealingTracker:
    def __init__(self, agent_ids):
        self._streak = {aid: 0 for aid in agent_ids}
        self._heal_count = {aid: 0 for aid in agent_ids}   # tracks repeated healing → escalate severity

    def update(self, agent_id, fitness):
        if fitness < HEAL_THRESHOLD:
            self._streak[agent_id] = self._streak.get(agent_id, 0) + 1
        else:
            self._streak[agent_id] = 0

    def needs_healing(self, agent_id):
        return self._streak.get(agent_id, 0) >= HEAL_STREAK

    def reset(self, agent_id):
        self._streak[agent_id] = 0


def heal(chrom, donor=None, severity=1, incident_related=False):
    base = crossover(chrom, donor) if donor is not None else deepcopy(chrom)

    if incident_related and severity == 1:
        base["epsilon"]  = float(np.random.uniform(0.15, 0.30))
        base["ent_coef"] = float(np.random.uniform(0.01, 0.05))
        return mutate(base, rate=0.15, scale=0.10)

    rate  = min(0.15 * severity, 0.50)
    scale = min(0.10 * severity, 0.20)
    base["epsilon"] = float(np.random.uniform(0.15, 0.30))
    base["lr"]      = float(np.random.uniform(1e-4, 1e-3))
    return mutate(base, rate=rate, scale=scale)


# ── generation step (GA only — PSO/GWO/Hybrid in evolution.py) ───────────────

def next_generation(population, fitnesses):
    ranked  = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
    elites  = [deepcopy(c) for _, c in ranked[:GA_ELITE_N]]
    offspring = []
    while len(offspring) < GA_POPULATION - GA_ELITE_N:
        offspring.append(mutate(crossover(
            tournament_select(population, fitnesses),
            tournament_select(population, fitnesses),
        )))
    return elites + offspring


# ── persistence ───────────────────────────────────────────────────────────────

def save_population(population, fitnesses, generation, path, genome_mode="shared"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"gen_{generation:04d}.json"), "w") as f:
        json.dump({"generation": generation,
                   "genome_mode": genome_mode,
                   "population": population,
                   "fitnesses": fitnesses}, f, indent=2)


def load_population(filepath):
    with open(filepath) as f:
        p = json.load(f)
    return p["population"], p["fitnesses"], p["generation"]
 
_fitness_cache = {}

def cached_fitness(chromosome: dict, compute_fn, scenario: str = "") -> float:
    key = hash((tuple(sorted(encode(chromosome).tolist())), scenario))
    if key not in _fitness_cache:
        _fitness_cache[key] = compute_fn()
    return _fitness_cache[key]

def clear_fitness_cache():
    _fitness_cache.clear()