import numpy as np
from copy import deepcopy
from ga import (
    encode, decode, GENE_KEYS, GA_ELITE_N,
    GA_MUTATION_RATE, GA_POPULATION,
    tournament_select, crossover, mutate,
)

# ── base class ────────────────────────────────────────────────────────────────

class BaseEvolver:
    def step(self, population: list, fitnesses: list) -> list:
        raise NotImplementedError

    def _elites(self, population, fitnesses):
        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        return [deepcopy(c) for _, c in ranked[:GA_ELITE_N]]

    def _best_idx(self, fitnesses):
        return int(np.argmax(fitnesses))


# ── GA ────────────────────────────────────────────────────────────────────────

class GAEvolver(BaseEvolver):
    def step(self, population, fitnesses):
        elites   = self._elites(population, fitnesses)
        offspring = []
        while len(offspring) < GA_POPULATION - GA_ELITE_N:
            p_a   = tournament_select(population, fitnesses)
            p_b   = tournament_select(population, fitnesses)
            child = mutate(crossover(p_a, p_b))
            offspring.append(child)
        return elites + offspring


# ── PSO ───────────────────────────────────────────────────────────────────────

class PSOEvolver(BaseEvolver):
    """
    Particle Swarm Optimisation.
    Each chromosome is a particle with a velocity.
    Converges fast toward known good regions.
    w  = inertia (momentum of current direction)
    c1 = cognitive (pull toward particle's own best)
    c2 = social    (pull toward swarm's global best)
    """
    def __init__(self, w=0.7, c1=1.2, c2=1.0):
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self._velocities  = None
        self._pbest       = None   # personal best per particle
        self._pbest_score = None
        self._gbest       = None   # global best

    def _init(self, population):
        n        = len(population)
        enc_size = len(encode(population[0]))
        self._velocities  = [np.zeros(enc_size) for _ in range(n)]
        self._pbest       = [encode(c) for c in population]
        self._pbest_score = [-np.inf] * n
        self._gbest       = self._pbest[0].copy()
        self._gbest_score = -np.inf

    def step(self, population, fitnesses):
        if self._velocities is None:
            self._init(population)

        # update personal and global bests
        for i, (f, c) in enumerate(zip(fitnesses, population)):
            if f > self._pbest_score[i]:
                self._pbest_score[i] = f
                self._pbest[i] = encode(c).copy()
            if f > self._gbest_score:
                self._gbest_score = f
                self._gbest = encode(c).copy()

        new_pop = []
        for i, chrom in enumerate(population):
            pos = encode(chrom)
            enc_size = len(encode(chrom))
            r1, r2 = np.random.rand(enc_size), np.random.rand(enc_size)
            self._velocities[i] = (
                self.w  * self._velocities[i]
                + self.c1 * r1 * (self._pbest[i] - pos)
                + self.c2 * r2 * (self._gbest    - pos)
            )
            new_pos = np.clip(pos + self._velocities[i], 0, 1)
            new_pop.append(decode(new_pos))

        # preserve elites
        elites = self._elites(population, fitnesses)
        new_pop[:GA_ELITE_N] = elites
        return new_pop


# ── GWO ───────────────────────────────────────────────────────────────────────

class GWOEvolver(BaseEvolver):
    """
    Grey Wolf Optimiser.
    Alpha = best solution, Beta = 2nd best, Delta = 3rd best.
    Wolves (chromosomes) hunt by converging toward alpha/beta/delta.
    Better at escaping local optima than plain PSO.
    a decreases linearly 2→0 over generations (controls exploration→exploitation).
    """
    def __init__(self, max_gen=50):
        self.max_gen  = max_gen
        self.gen      = 0

    def step(self, population, fitnesses):
        a = 2.0 - 2.0 * (self.gen / self.max_gen)   # linearly decreasing
        self.gen += 1

        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        alpha  = encode(ranked[0][1])
        beta   = encode(ranked[1][1]) if len(ranked) > 1 else alpha
        delta  = encode(ranked[2][1]) if len(ranked) > 2 else beta

        new_pop = []
        for chrom in population:
            pos      = encode(chrom)
            enc_size = len(pos)
            new_pos  = np.zeros(enc_size)

            for leader in (alpha, beta, delta):
                r1, r2 = np.random.rand(enc_size), np.random.rand(enc_size)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - pos)
                new_pos += leader - A * D

            new_pos = np.clip(new_pos / 3.0, 0, 1)
            new_pop.append(decode(new_pos))

        elites = self._elites(population, fitnesses)
        new_pop[:GA_ELITE_N] = elites
        return new_pop


# ── Hybrid GA + PSO + GWO ─────────────────────────────────────────────────────

class HybridEvolver(BaseEvolver):
    """
    Merges GA, PSO, and GWO into one generation step. // -> compared to singletons
    Each chromosome gets updated by all three, then the best result is kept.
    Population is split into thirds — each third gets a different primary driver,
    all three influence the final position via weighted blend.
    """
    def __init__(self, max_gen=50):
        self._ga  = GAEvolver()
        self._pso = PSOEvolver()
        self._gwo = GWOEvolver(max_gen=max_gen)

    def step(self, population, fitnesses, eval_fn=None):
         
        n = len(population)

        ga_pop  = self._ga.step(deepcopy(population), fitnesses)
        pso_pop = self._pso.step(deepcopy(population), fitnesses)
        gwo_pop = self._gwo.step(deepcopy(population), fitnesses)

        from config import HYBRID_SELECTION
        best_enc = encode(population[self._best_idx(fitnesses)])
        new_pop  = []
       
        for i in range(n):
            candidates = [ga_pop[i], pso_pop[i], gwo_pop[i]]
            if HYBRID_SELECTION == "performance" and eval_fn is not None:
                # real evaluation: actually test each candidate before picking
                scores = [eval_fn(c) for c in candidates]
                new_pop.append(deepcopy(candidates[int(np.argmax(scores))]))
            elif HYBRID_SELECTION == "proximity":
                # proximity: pick candidate closest to current global best
                dists = [np.linalg.norm(encode(c) - best_enc) for c in candidates]
                new_pop.append(deepcopy(candidates[int(np.argmin(dists))]))
            else:
                new_pop.append(deepcopy(candidates[np.random.randint(3)]))

        elites = self._elites(population, fitnesses)
        new_pop[:GA_ELITE_N] = elites
        return new_pop


# ── factory ───────────────────────────────────────────────────────────────────

def get_evolver(strategy: str, max_gen: int = 50) -> BaseEvolver:
    """
    strategy: "ga" | "pso" | "gwo" | "hybrid"
    Pass max_gen = GA_GENERATIONS from config for correct GWO decay schedule.
    """
    options = {
        "ga":     GAEvolver,
        "pso":    PSOEvolver,
        "gwo":    lambda: GWOEvolver(max_gen=max_gen),
        "hybrid": lambda: HybridEvolver(max_gen=max_gen),
    }
    if strategy not in options:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(options)}")
    cls = options[strategy]
    return cls() if callable(cls) and not isinstance(cls, type) else cls()