import numpy as np
from collections import defaultdict
from config import ALPHA, GAMMA
from ga import bellman_target

DECAY = 0.9


class Coordinator:
    def __init__(self, agent_ids: list):
        self.agent_ids   = agent_ids
        self._data       = defaultdict(lambda: {"wait": [], "real_wait": [], "queue": [], "passed": [], "stops": []})
        self._phero      = {aid: 0.0 for aid in agent_ids}
        self._outflow    = {aid: 0.0 for aid in agent_ids}
        self._value_est  = {aid: 0.0 for aid in agent_ids}
        self._value_nets = {}   
        self._alpha       = ALPHA    
        self._gamma_shape = GAMMA 
        self._ep_rewards = {aid: [] for aid in agent_ids}  # for TD(λ) # V(s) estimate per agent
        self._neighbours = {}
        self._downstream_space  = {aid: 1.0 for aid in agent_ids}
        self._incident_flags    = {aid: 0.0 for aid in agent_ids}
        self._incident_occurred = False
        self._last_arrival = 0.0   # tracks network clearance time
        self._peak_vehicles = 0    # tracks worst congestion moment

    def reset(self):
        self._data      = defaultdict(lambda: {"wait": [], "real_wait": [], "queue": [], "passed": [], "stops": []})
        self._phero     = {aid: 0.0 for aid in self.agent_ids}
        self._outflow   = {aid: 0.0 for aid in self.agent_ids}
        self._value_est  = {aid: 0.0 for aid in self.agent_ids}
        self._downstream_space = {aid: 1.0 for aid in self.agent_ids}
        self._incident_flags = {aid: 0.0 for aid in self.agent_ids}
        self._last_arrival = 0.0
        self._peak_vehicles = 0
        self._ep_rewards = {aid: [] for aid in self.agent_ids}

    # ── pheromone ─────────────────────────────────────────────────────────────
    def update_pheromones(self, queue_dict: dict):
        for aid in self.agent_ids:
            self._phero[aid] = self._phero[aid] * DECAY + queue_dict.get(aid, 0.0)

    def pheromone_map(self) -> dict:
        return dict(self._phero)

    def neighbour_pheromone(self, agent_id: str, neighbours: list = None, hops: int = 2) -> float:
        visited, frontier, signal, decay = {agent_id}, (neighbours or self._neighbours.get(agent_id, [])), 0.0, 1.0
        for _ in range(hops):
            frontier = [n for n in frontier if n not in visited]
            if not frontier:
                break
            weights = [1/(i+1) for i in range(len(frontier))]
            total = sum(weights) or 1.0
            signal += decay * sum((w/total) * self._phero.get(n, 0.0) for w, n in zip(weights, frontier))
            visited.update(frontier)
            frontier = [nn for n in frontier for nn in self._neighbours.get(n, [])]
            decay *= DECAY
        return signal

    # ── downstream outflow ────────────────────────────────────────────────────

    def update_outflow(self, outflow_dict: dict):
        for aid, count in outflow_dict.items():
            self._outflow[aid] = float(count)

    def incoming_signal(self, agent_id: str) -> float:
        nbrs = self._neighbours.get(agent_id, [])
        return sum(self._outflow.get(n, 0.0) for n in nbrs)

    def update_incidents(self, incident_dict: dict):
        self._incident_flags = incident_dict
        if any(v > 0 for v in incident_dict.values()):
            self._incident_occurred = True

    def incident_signal(self, agent_id: str) -> float:
        nbrs = self._neighbours.get(agent_id, [])
        return max((self._incident_flags.get(n, 0.0) for n in nbrs), default=0.0)

    # ── Bellman-bootstrapped reward shaping ───────────────────────────────────
    def shaped_rewards(self, raw_rewards: dict, baseline: float = None, h_t_dict: dict = None) -> dict:
        """
        TD(λ) + advantage + reward normalisation.
        Normalises by baseline wait so peak/offpeak/weekend are comparable.
        A(s,a) = bellman_r - V(s) — advantage over baseline value.
        """
        global_bonus = self._global_bonus()
        # normalise rewards so scenarios are comparable
        if baseline and baseline > 0:
            norm_factor = 1.0 / baseline
        else:
            norm_factor = 1.0
        shaped = {}
        for aid, r in raw_rewards.items():
            r_norm           = r * norm_factor
            v_s              = self._value_est.get(aid, 0.0)
            v_s_next         = r_norm

            if aid in self._value_nets and self._value_nets[aid] is not None:
                import torch
                net = self._value_nets[aid]
                device = next(net.parameters()).device
                if h_t_dict is not None and aid in h_t_dict:
                    h = h_t_dict[aid].unsqueeze(0) if h_t_dict[aid].dim() == 1 else h_t_dict[aid]
                    h = h.to(device)
                else:
                    h = torch.zeros(1, 64, device=device)   # fallback if no hidden state supplied
                v_s_next = net.forward(h).item()
                from ga import bellman_target
                target   = bellman_target(r_norm, v_s_next, self._gamma_shape)
                net.update(h, target)
            bellman_r        = bellman_target(r_norm, v_s_next, self._gamma_shape)
            advantage        = bellman_r - v_s
            shaped[aid]      = advantage + global_bonus
            self._value_est[aid] = v_s_next
            self._ep_rewards[aid].append(r_norm)
        return shaped

    def _global_bonus(self) -> float:
        total_passed = sum(sum(d["passed"]) for d in self._data.values())
        return self._alpha * total_passed / max(len(self.agent_ids), 1)

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, agent_id: str, local_wait: float, queue_len: float,
               vehicles_passed: int, stops: float = 0.0, real_wait: float = None):
        self._data[agent_id]["wait"].append(local_wait)
        self._data[agent_id]["real_wait"].append(real_wait if real_wait is not None else 0.0)
        self._data[agent_id]["queue"].append(queue_len)
        self._data[agent_id]["passed"].append(vehicles_passed)
        self._data[agent_id]["stops"].append(stops)
        if self.incident_signal(agent_id) > 0:
            self._data[agent_id].setdefault("wait_during_incident", []).append(local_wait)

    # ── episode summary ───────────────────────────────────────────────────────
    def episode_summary(self) -> dict:
        summary = {}
        for aid in self.agent_ids:
            d = self._data[aid]
            summary[aid] = {
                "mean_wait":       np.mean(d["real_wait"]) if d["real_wait"] else 0.0,
                "reward_signal":   np.mean(d["wait"])      if d["wait"]      else 0.0,
                "max_queue":       np.max(d["queue"])  if d["queue"] else 0.0,
                "avg_stopped_cars": np.mean(d["queue"]) if d["queue"] else 0.0,
                "total_passed":    int(np.sum(d["passed"])),
                "total_stops":     float(np.sum(d["stops"])),
                "overflow_events": int(np.sum(np.array(d["queue"]) > 20)),
                "mean_wait_during_incident": float(np.mean(d["wait_during_incident"])) if d.get("wait_during_incident") else None,
            }
        all_waits = [v for d in self._data.values() for v in d["real_wait"]]
        all_reward_signal = [v for d in self._data.values() for v in d["wait"]]
        summary["__network__"] = {
            "mean_wait":    np.mean(all_waits) if all_waits else 0.0,
            "reward_signal": np.mean(all_reward_signal) if all_reward_signal else 0.0,
            "std_wait":   np.std(all_waits)  if all_waits else 0.0,

            "total_passed": sum(s["total_passed"] for s in summary.values()
                   if isinstance(s, dict)),
            "vehicles_per_hour": sum(s["total_passed"] for s in summary.values()
                        if isinstance(s, dict)) / max(len(all_waits), 1) * 3600,

            "network_clearance_time": self._last_arrival,
            "peak_congestion": self._peak_vehicles,
 
            "per_agent_waits": {
                aid: np.mean(self._data[aid]["real_wait"])
                for aid in self.agent_ids if self._data[aid]["real_wait"]
            },
        }
        from ga import td_lambda_returns
        for aid in self.agent_ids:
            if self._ep_rewards.get(aid):
                summary[aid]["td_lambda_return"] = float(
                    np.mean(td_lambda_returns(self._ep_rewards[aid]))
                )
        return summary        

    def update_last_arrival(self, current_time: float, vehicles_present: int):
        if vehicles_present > 0:
            self._last_arrival = current_time
        self._peak_vehicles = max(self._peak_vehicles, vehicles_present)

    def congestion_map(self) -> dict:
        return {
            aid: np.mean(self._data[aid]["queue"]) if self._data[aid]["queue"] else 0.0
            for aid in self.agent_ids
        }

    # ── gene sharing (GENOME_MODE == "per_agent" only) ─────────────────────────
    def share_genes(self, populations: dict, rate: float = 0.15,
                     target: str = "last", fitnesses: dict = None) -> dict:
        from ga import crossover
        for aid, nbrs in self._neighbours.items():
            if not nbrs or aid not in populations:
                continue
            if np.random.rand() < rate:
                donor = nbrs[np.random.randint(len(nbrs))]
                if donor not in populations or not populations[donor]:
                    continue
                best_self  = populations[aid][0]
                best_donor = populations[donor][0]
                if target == "weakest" and fitnesses and aid in fitnesses and fitnesses[aid]:
                    replace_idx = int(np.argmin(fitnesses[aid]))
                else:
                    replace_idx = -1
                populations[aid][replace_idx] = crossover(best_self, best_donor)
        return populations
    
    def write_live_state(self, step: int, generation: int,
                     strategy: str, healing: list):
        """Writes current state to file for visualiser to read."""
        import json, os
        agents_data = {
            aid: {
                "queue": float(np.mean(self._data[aid]["queue"][-10:]))
                        if self._data[aid]["queue"] else 0.0,
                "wait":  float(np.mean(self._data[aid]["wait"][-10:]))
                        if self._data[aid]["wait"] else 0.0,
            }
            for aid in self.agent_ids
        }
        payload = {
            "step":       step,
            "generation": generation,
            "strategy":   strategy,
            "mean_wait":  float(np.mean([v for d in self._data.values() for v in d["wait"]]) 
              if any(d["wait"] for d in self._data.values()) else 0.0),
            "agents":     agents_data,
            "pheromones": dict(self._phero),
            "incoming":   dict(self._outflow),
            "blocked":    [],
            "healing":    healing,
        }
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/live_state.json", "w") as f:
            json.dump(payload, f)