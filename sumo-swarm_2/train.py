import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import (ROUTE_FILES, CHECKPOINT_DIR, SAVE_EVERY_N_GEN,
                    GA_POPULATION, GA_GENERATIONS, SCENARIO_WEIGHTS,
                    POPULATION_RESTART_EVERY, GENOME_MODE, GENE_SHARE_RATE,
                    GENE_SHARE_EVERY, HYBRID_SELECTION, NETWORK_PROFILE,
                    RESTART_MODE, RESTART_STAGNATION_GENS, GENE_SHARE_TARGET, ALPHA, GAMMA)
from env         import SumoSwarmEnv
from agent       import (make_agent, predict, predict_with_hidden,
                         init_for_training, predict_train, store_and_maybe_train)
from ga          import (random_chromosome, compute_fitness, fairness_penalty,
                         save_population, load_population, HealingTracker, heal)
from evolution   import get_evolver
from coordinator import Coordinator
from utils       import set_seed, RunLogger, save_agent


def run_episode(env, agents, coordinator, gen=0, strategy="hybrid") -> dict:
    try:
        import libsumo as traci
    except ImportError:
        import traci
    

    obs  = env.reset()
    coordinator.reset()
    done = False
    is_first_step = True

    while not done:
        actions, h_t_dict, values, log_probs = {}, {}, {}, {}
        for aid in env.agent_ids:
            action, value, log_prob, h_t = predict_train(agents[aid], obs[aid])
            actions[aid]    = action
            h_t_dict[aid]   = h_t
            values[aid]     = value
            log_probs[aid]  = log_prob
        prev_obs = obs
        obs, rewards, done, _ = env.step(actions)

        shaped = coordinator.shaped_rewards(rewards, h_t_dict=h_t_dict)
        for aid in env.agent_ids:
            store_and_maybe_train(
                agents[aid], prev_obs[aid], actions[aid], shaped[aid],
                is_first_step, values[aid], log_probs[aid], obs[aid],
            )
        is_first_step = False

        network_arrived = traci.simulation.getArrivedNumber()
        for aid in env.agent_ids:
            lanes  = traci.trafficlight.getControlledLanes(aid)
            q      = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            passed = network_arrived / max(len(env.agent_ids), 1)
            stops  = float(sum(traci.lane.getLastStepHaltingNumber(l) > 0
                         for l in lanes))
            real_wait = sum(traci.lane.getWaitingTime(l) for l in lanes) / max(q, 1)
            coordinator.record(aid, -rewards[aid], q, passed, stops, real_wait=real_wait)

        if hasattr(coordinator, "write_live_state"):
            coordinator.write_live_state(
                step=env.step_count,
                generation=gen,
                strategy=strategy,
                healing=[],
            )

    return coordinator.episode_summary()

def evaluate_chromosome(env, coordinator, agent_ids, chromosome, n_episodes=3, tag="") -> float:
    """Run a chromosome across peak/offpeak/weekend, return mean fitness.
    Used by HybridEvolver to actually test GA/PSO/GWO candidates."""
    print(f"    ↳ evaluating hybrid candidate {tag}...", flush=True)
    
    agents = {aid: init_for_training(make_agent(chromosome)) for aid in agent_ids}
    env._phase_hold = chromosome.get("phase_hold", 5.0)
    env._nbr_radius = chromosome.get("nbr_radius", 2)
    coordinator._alpha = chromosome.get("alpha", ALPHA)
    coordinator._gamma_shape = chromosome.get("gamma", GAMMA)
    route_names = list(ROUTE_FILES.keys())
    
    scores = []
    for ep in range(n_episodes):
        scenario = route_names[ep % len(route_names)]
        env.route_file = ROUTE_FILES[scenario]
        summary = run_episode(env, agents, coordinator, strategy="hybrid")
        net = summary["__network__"]
        fp = fairness_penalty(net.get("per_agent_waits", {}))
        total_stops = sum(summary[a]["total_stops"] for a in agent_ids if a in summary)
        fitness = compute_fitness(
            wait_std=net.get("std_wait", 0.0),
            local_wait=net["mean_wait"],
            queue_overflows=sum(summary[a]["overflow_events"] for a in agent_ids if a in summary),
            network_throughput=net["total_passed"],
            stops=total_stops,
            fairness_penalty=fp,
            chromosome=chromosome,
        )
        scores.append(fitness * SCENARIO_WEIGHTS.get(scenario, 1.0))
    return float(np.mean(scores))

def save_per_agent_checkpoint(population: dict, fitnesses: dict, generation: int, path: str):
     
    import json
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"gen_{generation:04d}.json"), "w") as f:
        json.dump({
            "generation":  generation,
            "genome_mode": "per_agent",
            "population":  population,   # {aid: [chrom, ...]}
            "fitnesses":   fitnesses,    # {aid: [fitness, ...]}
        }, f, indent=2)


def run_per_agent_generation(args, env, coordinator, agent_ids, population,
                              evolvers, fitnesses, healer, ckpt_dir, logger,
                              start_gen):
    
    from agent import make_value_net
    route_names = list(ROUTE_FILES.keys())
    agents = {}
    aid_elites = {}
    _best_fit_ever = {aid: -np.inf for aid in agent_ids}   # used only when RESTART_MODE == "fitness"
    _stall_count   = {aid: 0 for aid in agent_ids}

    for gen in range(start_gen, args.generations):
        print(f"\n── [{args.evolution.upper()} | per_agent] Generation {gen+1}/{args.generations} ──")
        gen_fitnesses = {aid: [] for aid in agent_ids}
        best_agents_this_gen  = {aid: None for aid in agent_ids}
        best_fitness_this_gen = {aid: -np.inf for aid in agent_ids}

        for pop_idx in range(GA_POPULATION):
            if pop_idx == 0 and gen > start_gen and aid_elites:
                agents = {aid: aid_elites[aid] for aid in agent_ids if aid in aid_elites}
                for aid in agent_ids:
                    if aid not in agents:
                        agents[aid] = init_for_training(make_agent(population[aid][pop_idx]))
            else:
                agents = {aid: init_for_training(make_agent(population[aid][pop_idx])) for aid in agent_ids}
            if pop_idx == 0:
                elite_agents_this_gen = dict(agents)
            value_nets = {aid: make_value_net() for aid in agent_ids}
            coordinator._value_nets = value_nets
            env._phase_hold = float(np.median(
                [population[aid][pop_idx].get("phase_hold", 5.0) for aid in agent_ids]
            ))
            coordinator._alpha = float(np.median(
                [population[aid][pop_idx].get("alpha", ALPHA) for aid in agent_ids]
            ))
            coordinator._gamma_shape = float(np.median(
                [population[aid][pop_idx].get("gamma", GAMMA) for aid in agent_ids]
            ))
            env._nbr_radius = int(round(np.median([population[aid][pop_idx].get("nbr_radius", 2) for aid in agent_ids])))
            
            ep_summaries = []
            for ep in range(args.episodes):
                scenario = route_names[ep % len(route_names)]
                env.route_file = ROUTE_FILES[scenario]
                summary = run_episode(env, agents, coordinator, gen=gen,
                                       strategy=f"{args.evolution}_per_agent")
                ep_summaries.append((summary, scenario))

            for aid in agent_ids:
                scores = []
                for ep_idx, (summary, scenario) in enumerate(ep_summaries):
                    if aid not in summary:
                        continue
                    a = summary[aid]
                    fitness = compute_fitness(
                        wait_std=0.0,
                        local_wait=a["mean_wait"],
                        queue_overflows=a["overflow_events"],
                        network_throughput=a["total_passed"],
                        stops=a["total_stops"],
                        fairness_penalty=0.0,
                        chromosome=population[aid][pop_idx],
                    )
                    weighted = fitness * SCENARIO_WEIGHTS.get(scenario, 1.0)
                    weighted += 0.1 * a.get("td_lambda_return", 0.0)
                    scores.append(weighted)
                    logger.log({
                        "strategy":   f"{args.evolution}_per_agent",
                        "agent_id":   aid,
                        "generation": gen,
                        "pop_idx":    pop_idx,
                        "episode":    ep_idx,
                        "scenario":   scenario,
                        "fitness":    round(weighted, 4),
                        "mean_wait":  round(a["mean_wait"], 2),
                    })
                gen_fitnesses[aid].append(float(np.mean(scores)) if scores else 0.0)
                if gen_fitnesses[aid][-1] > best_fitness_this_gen[aid]:
                    best_fitness_this_gen[aid] = gen_fitnesses[aid][-1]
                    best_agents_this_gen[aid]  = agents[aid]

            print(f"  [{pop_idx+1}/{GA_POPULATION}] evaluated across {len(agent_ids)} junctions")

        for aid in agent_ids:
            fitnesses[aid] = gen_fitnesses[aid]
            aid_elites[aid] = best_agents_this_gen[aid] or elite_agents_this_gen[aid]

            gen_best = max(gen_fitnesses[aid]) if gen_fitnesses[aid] else -np.inf
            if gen_best > _best_fit_ever[aid]:
                _best_fit_ever[aid] = gen_best
                _stall_count[aid] = 0
            else:
                _stall_count[aid] += 1

            healer.update(aid, float(np.mean(gen_fitnesses[aid])))
            if coordinator._incident_occurred and np.mean(gen_fitnesses[aid]) < 0:
                import config as _cfg
                boosted_agent = elite_agents_this_gen[aid]
                boosted_agent.ent_coef = min(boosted_agent.ent_coef * 1.5, 0.05)
                boosted_agent._ent_coef_boost_until = boosted_agent.num_timesteps + _cfg.EXPECTED_CANDIDATE_STEPS

            if healer.needs_healing(aid):
                healer._heal_count[aid] = healer._heal_count.get(aid, 0) + 1
                severity = healer._heal_count[aid]
                nbrs = coordinator._neighbours.get(aid, [])
                donor = None
                if nbrs:
                    donor_aid = max(nbrs, key=lambda n: np.mean(fitnesses.get(n, [0.0])))
                    if population.get(donor_aid):
                        donor = population[donor_aid][0]
                print(f"  ⚠ healing agent {aid} (severity={severity}, donor={donor_aid if donor else 'none'})")
                population[aid] = [
                    heal(c, donor=donor, severity=severity,
                         incident_related=coordinator._incident_occurred)
                    for c in population[aid]
                ]
                healer.reset(aid)
            else:
                healer._heal_count[aid] = 0   # reset escalation once agent recovers on its own
            # population[aid] = evolvers[aid].step(population[aid], fitnesses[aid])
            if args.evolution == "hybrid" and HYBRID_SELECTION == "performance":
                def eval_fn(candidate, _aid=aid, _counter=[0]):
                    _counter[0] += 1
                    print(f"    ↳ probing {_aid} candidate {_counter[0]}/3", flush=True)
                    
                    """Probe one candidate for _aid only; every other junction stays
                    at its current elite chromosome. 1 scenario/episode to control cost."""
                    probe_agents = {a: init_for_training(make_agent(population[a][0])) for a in agent_ids if a != _aid}
                    probe_agents[_aid] = init_for_training(make_agent(candidate))

                    env._phase_hold = candidate.get("phase_hold", 5.0)
                    coordinator._alpha = candidate.get("alpha", ALPHA)
                    coordinator._gamma_shape = candidate.get("gamma", GAMMA)
                    env._nbr_radius = candidate.get("nbr_radius", 2)
                    env.route_file  = ROUTE_FILES[route_names[0]]
                    summary = run_episode(env, probe_agents, coordinator, gen=gen,
                                           strategy="hybrid_per_agent_probe")
                    a = summary.get(_aid, {})
                    return compute_fitness(
                        wait_std=0.0, local_wait=a.get("mean_wait", 0.0),
                        queue_overflows=a.get("overflow_events", 0),
                        network_throughput=a.get("total_passed", 0),
                        stops=a.get("total_stops", 0), fairness_penalty=0.0,
                        chromosome=candidate,
                    )
                population[aid] = evolvers[aid].step(population[aid], fitnesses[aid], eval_fn=eval_fn)
            else:
                population[aid] = evolvers[aid].step(population[aid], fitnesses[aid])

        if (gen + 1) % GENE_SHARE_EVERY == 0:
            population = coordinator.share_genes(population, rate=GENE_SHARE_RATE,
                                                   target=GENE_SHARE_TARGET, fitnesses=fitnesses)
            print(f"  ↔ neighbour gene-sharing applied (target={GENE_SHARE_TARGET})")

        n_inject = 2
        for aid in agent_ids:
            if RESTART_MODE == "fitness":
                trigger = _stall_count[aid] >= RESTART_STAGNATION_GENS
            else:
                trigger = (gen + 1) % POPULATION_RESTART_EVERY == 0
            if not trigger:
                continue
            ranked_idx = sorted(range(len(population[aid])),
                                 key=lambda i: fitnesses[aid][i])
            for i in ranked_idx[:n_inject]:
                population[aid][i] = random_chromosome()
            if RESTART_MODE == "fitness":
                _stall_count[aid] = 0   # reset the clock after injecting
            print(f"  ↺ injected {n_inject} random chromosomes for {aid} (mode={RESTART_MODE})")

        if (gen + 1) % SAVE_EVERY_N_GEN == 0:
        
            save_per_agent_checkpoint(population, fitnesses, gen + 1, ckpt_dir)
            for aid in agent_ids:
                save_agent(best_agents_this_gen[aid], os.path.join(ckpt_dir, "agents", aid, "best_agent"))
            print(f"  checkpoint → {ckpt_dir}/gen_{gen+1:04d}.json")

    return population


def train(args):
    set_seed(42)
    import config as _cfg
    _cfg.EXPECTED_CANDIDATE_STEPS = args.episodes * _cfg.EPISODE_STEPS

    #ckpt_dir = os.path.join(CHECKPOINT_DIR, args.evolution)
    from config import HYBRID_SELECTION
    strategy_label = f"{args.evolution}_{HYBRID_SELECTION}" if args.evolution == "hybrid" else args.evolution
    if GENOME_MODE == "per_agent":
        strategy_label += "_per_agent"
    strategy_label += f"_{NETWORK_PROFILE}"
    ckpt_dir = os.path.join(CHECKPOINT_DIR, strategy_label)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    logger  = RunLogger(log_dir=ckpt_dir, use_wandb=args.wandb)
    evolver = get_evolver(args.evolution, max_gen=args.generations)

    env         = SumoSwarmEnv(route_file=ROUTE_FILES["peak"])
    env.reset()
    agent_ids   = env.agent_ids
    coordinator = Coordinator(agent_ids)
    env.coordinator = coordinator
    route_cycle = list(ROUTE_FILES.values())   # [peak, offpeak, weekend]
    healer      = HealingTracker(agent_ids)
    evolvers    = None

    if args.resume:
        population, fitnesses, start_gen = load_population(args.resume)
        if GENOME_MODE == "per_agent":
            evolvers = {aid: get_evolver(args.evolution, max_gen=args.generations) for aid in agent_ids}
        print(f"Resumed {args.evolution} from generation {start_gen}")
    elif GENOME_MODE == "per_agent":
        population = {aid: [random_chromosome() for _ in range(GA_POPULATION)]
                      for aid in agent_ids}
        fitnesses  = {aid: [0.0] * GA_POPULATION for aid in agent_ids}
        evolvers   = {aid: get_evolver(args.evolution, max_gen=args.generations)
                      for aid in agent_ids}
        start_gen  = 0
    else:
        population  = [random_chromosome() for _ in range(GA_POPULATION)]
        fitnesses   = [0.0] * GA_POPULATION
        start_gen   = 0

    if GENOME_MODE == "per_agent":
        population = run_per_agent_generation(args, env, coordinator, agent_ids,
                                               population, evolvers, fitnesses,
                                               healer, ckpt_dir, logger, start_gen)
        env.close()
        logger.close()
        print(f"\n{args.evolution.upper()} (per_agent) training complete.")
        return

    _best_fit_ever = -np.inf   
    _stall_count   = 0

    for gen in range(start_gen, args.generations):
        print(f"\n── [{args.evolution.upper()}] Generation {gen+1}/{args.generations} ──")
        from ga import clear_fitness_cache
        clear_fitness_cache()
        gen_fitnesses = []
        best_agents_this_gen  = None
        best_fitness_this_gen = -np.inf

        for pop_idx, chromosome in enumerate(population):
        
            agents      = {aid: init_for_training(make_agent(chromosome)) for aid in agent_ids}
            env._phase_hold = chromosome.get("phase_hold", 5.0)
            env._nbr_radius = chromosome.get("nbr_radius", 2)
            coordinator._alpha = chromosome.get("alpha", ALPHA)
            coordinator._gamma_shape = chromosome.get("gamma", GAMMA)
            from agent import make_value_net
            
            value_nets  = {aid: make_value_net() for aid in agent_ids}
            coordinator._value_nets = value_nets
            ep_fitnesses = []

            route_names = list(ROUTE_FILES.keys())   # peak, offpeak, weekend
            for ep in range(args.episodes):
                scenario   = route_names[ep % len(route_names)]
                env.route_file = ROUTE_FILES[scenario]
                summary = run_episode(env, agents, coordinator, gen=gen, strategy=args.evolution)
                net     = summary["__network__"]

                fp = fairness_penalty(net.get("per_agent_waits", {}))
                total_stops = sum(
                    summary[a]["total_stops"]
                    for a in agent_ids if a in summary
                )
                from ga import cached_fitness
                raw_fitness = cached_fitness(
                        chromosome,
                        lambda: compute_fitness(
                        wait_std=net.get("std_wait", 0.0),
                        local_wait=net["mean_wait"],
                        queue_overflows=sum(summary[a]["overflow_events"]
                                        for a in agent_ids if a in summary),
                        network_throughput=net["total_passed"],
                        stops=total_stops,
                        fairness_penalty=fp,
                        chromosome=chromosome,
                    ),
                    scenario=scenario,
                )
                # weight by scenario importance
                weight  = SCENARIO_WEIGHTS.get(scenario, 1.0)
                fitness = raw_fitness * weight
                fitness += 0.1 * np.mean([
                    summary[a].get("td_lambda_return", 0.0)
                    for a in agent_ids if a in summary
                ])
                ep_fitnesses.append(fitness)
                logger.log({
                    "strategy":        args.evolution,
                    "generation":      gen,
                    "pop_idx":         pop_idx,
                    "episode":         ep,
                    "scenario":        scenario,
                    "fitness":         round(fitness, 4),
                    "mean_wait":       round(net["mean_wait"], 2),
                    "fairness_penalty":round(fp, 4),
                })

            mean_fitness = float(np.mean(ep_fitnesses))
            gen_fitnesses.append(mean_fitness)
            if mean_fitness > best_fitness_this_gen:
                best_fitness_this_gen = mean_fitness
                best_agents_this_gen  = agents
            print(f"  [{pop_idx+1}/{GA_POPULATION}] fitness={mean_fitness:.4f}")

            for aid in agent_ids:
                healer.update(aid, mean_fitness)
                if healer.needs_healing(aid):
                    print(f"  ⚠ healing agent {aid}")
                    population[pop_idx] = heal(population[pop_idx])
                    healer.reset(aid)

        fitnesses  = gen_fitnesses

        gen_best = max(gen_fitnesses) if gen_fitnesses else -np.inf
        if gen_best > _best_fit_ever:
            _best_fit_ever = gen_best
            _stall_count = 0
        else:
            _stall_count += 1

        if args.evolution == "hybrid" and HYBRID_SELECTION == "performance":
            print(" NOW evaluating 3x candidates per chromosome (this takes a while)...")
            counter = {"n": 0}
            def eval_fn(chrom):
                counter["n"] += 1
                return evaluate_chromosome(env, coordinator, agent_ids, chrom, n_episodes=3, tag=f"{counter['n']}/18")
            population = evolver.step(population, fitnesses, eval_fn=eval_fn)
        else:
            population = evolver.step(population, fitnesses)
 
        if RESTART_MODE == "fitness":
            trigger = _stall_count >= RESTART_STAGNATION_GENS
        else:
            trigger = (gen + 1) % POPULATION_RESTART_EVERY == 0
        if trigger:
            n_inject = 2
            ranked_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
            for i in ranked_idx[:n_inject]:
                population[i] = random_chromosome()   # replace actual worst performers by fitness
            if RESTART_MODE == "fitness":
                _stall_count = 0   # reset the clock after injecting
            print(f"  ↺ injected {n_inject} random chromosomes (mode={RESTART_MODE})")


        if (gen + 1) % SAVE_EVERY_N_GEN == 0:
        
            save_population(population, fitnesses, gen + 1, ckpt_dir, genome_mode=GENOME_MODE)
            for aid in agent_ids:
                save_agent(best_agents_this_gen[aid], os.path.join(ckpt_dir, "agents", aid, "best_agent"))
            print(f"  checkpoint → {ckpt_dir}/gen_{gen+1:04d}.json")
            

    env.close()
    logger.close()
    print(f"\n{args.evolution.upper()} training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolution",   type=str, default="hybrid",
                        choices=["ga", "pso", "gwo", "hybrid"])
    parser.add_argument("--episodes",    type=int, default=3)
    parser.add_argument("--generations", type=int, default=GA_GENERATIONS)
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--resume",      type=str, default=None)
    train(parser.parse_args())