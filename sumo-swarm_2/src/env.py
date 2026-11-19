import os
import random
import numpy as np
from collections import deque
try:
    import libsumo as traci
    USING_LIBSUMO = True
except ImportError:
    import traci
    USING_LIBSUMO = False

from config import (
    SUMO_HOME, NET_FILE, EPISODE_STEPS,
    SIM_STEP_LENGTH, OBS_DIM, SEQ_LEN, USE_GUI, MAX_PHASES,
)

INCIDENT_PROB  = 0.02
INCIDENT_LANES = []


def _sumo_cmd(route_file: str, gui: bool = USE_GUI, seed: int = None) -> list:
    if os.name == "nt":
        binary = "sumo-gui.exe" if gui else "sumo.exe"
    else:
        binary = "sumo-gui" if gui else "sumo"
    return [
        os.path.join(SUMO_HOME, "bin", binary),
        "-n", NET_FILE,
        "-r", route_file,
        "--step-length", str(SIM_STEP_LENGTH),
        "--no-warnings",
        "--no-step-log",
        "--ignore-route-errors",
        "--waiting-time-memory", "1000",
    ]+ (["--seed", str(seed)] if seed is not None else [])


class SumoSwarmEnv:
    def __init__(self, route_file: str, coordinator=None, seed: int = None):
        self.route_file   = route_file
        self.coordinator  = coordinator   # injected — optional

        self.seed         = seed
        
        self.step_count   = 0
        self.tls_ids      = []
        self._obs_buffer  = {}
        self._blocked_lanes = {}
        self._neighbours  = {}  # {tls_id: [neighbour_ids]}
        self._phase_hold  = 5.0   # updated per chromosome each episode
        self._nbr_radius  = 2     # updated per chromosome each episode
        self.force_incidents = False   # set True from train.py for training runs

        self._obs_mean = None
        self._obs_std  = None
        self._obs_m2   = None   # Welford's accumulator for running variance
        self._obs_n    = 0

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        global INCIDENT_LANES
        if USING_LIBSUMO:
            try:
                traci.close()
            except Exception:
                pass
            traci.start([
                "sumo",
                "-n", NET_FILE,
                "-r", self.route_file,
                "--step-length", str(SIM_STEP_LENGTH),
                "--no-warnings",
                "--no-step-log",
                "--ignore-route-errors",
                "--waiting-time-memory", "1000",
            ] + (["--seed", str(self.seed)] if self.seed is not None else []))
        else:
            if traci.isLoaded():
                traci.close()
            traci.start(_sumo_cmd(self.route_file, seed=self.seed))


        self.tls_ids = [
            t for t in traci.trafficlight.getIDList()
            if len(traci.trafficlight.getAllProgramLogics(t)[0].phases) > 1
        ]
        self.step_count = 0
        self._blocked_lanes = {}

        INCIDENT_LANES = [
            lane for tls in self.tls_ids
            for lane in traci.trafficlight.getControlledLanes(tls)
        ]

        self._forced_incident_step = (
            random.randint(100, 400) if self.force_incidents else None
        )

        self._obs_buffer = {
            tls: deque([np.zeros(OBS_DIM)] * SEQ_LEN, maxlen=SEQ_LEN)
            for tls in self.tls_ids
        }
        # reset running obs stats each episode so peak/offpeak/weekend don't bleed together
        self._obs_mean = None
        self._obs_std  = None
        self._obs_m2   = None
        self._obs_n    = 0

        self._neighbours = self._build_neighbour_map()

        if self.coordinator:
            self.coordinator.reset()
            self.coordinator._neighbours = self._neighbours

        return self._get_all_obs()

    def step(self, actions: dict) -> tuple:
        self._apply_actions(actions)
        self._maybe_trigger_incident()
        self._clear_old_incidents()

        # push queue snapshot to coordinator for pheromone update
        if self.coordinator:
            queue_now = {
                tls: sum(traci.lane.getLastStepHaltingNumber(l)
                         for l in traci.trafficlight.getControlledLanes(tls))
                for tls in self.tls_ids
            }
            self.coordinator.update_pheromones(queue_now)
            outflow_now = {
                tls: sum(traci.lane.getLastStepVehicleNumber(l)
                        for l in traci.trafficlight.getControlledLanes(tls))
                for tls in self.tls_ids
            }
            self.coordinator.update_outflow(outflow_now) 
            incident_now = {
                tls: 1.0 if any(l in self._blocked_lanes for l in traci.trafficlight.getControlledLanes(tls)) else 0.0
                for tls in self.tls_ids
            }
            self.coordinator.update_incidents(incident_now)

            # track network clearance time
            vehicles_present = traci.simulation.getMinExpectedNumber()
            self.coordinator.update_last_arrival(
                traci.simulation.getTime(), vehicles_present
            )

            # green wave hint — prepare downstream junction
            self._coordinate_green_wave()
            # max-pressure check — update downstream space signal
            self._update_max_pressure()           

        traci.simulationStep()
        self.step_count += 1

        obs     = self._get_all_obs()
        rewards = self._get_all_rewards()
        done    = self.step_count >= EPISODE_STEPS
        info    = {"step": self.step_count, "blocked": list(self._blocked_lanes.keys())}

        return obs, rewards, done, info

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    @property
    def agent_ids(self) -> list:
        return list(self.tls_ids)

    # ── incident simulation ───────────────────────────────────────────────────
 
    def _maybe_trigger_incident(self):
        forced = (self._forced_incident_step is not None
                  and self.step_count == self._forced_incident_step)
        if INCIDENT_LANES and (random.random() < INCIDENT_PROB or forced):
            lane = random.choice(INCIDENT_LANES)
            if lane not in self._blocked_lanes:
                traci.lane.setAllowed(lane, [])
                self._blocked_lanes[lane] = self.step_count
                # broadcast closure to coordinator so neighbours see pressure spike
                if self.coordinator:
                    affected_tls = next(
                        (tls for tls in self.tls_ids
                        if lane in traci.trafficlight.getControlledLanes(tls)), None
                    )
                    if affected_tls:
                        spike = {tls: 0.0 for tls in self.tls_ids}
                        queue_now = sum(traci.lane.getLastStepHaltingNumber(l)
                                        for l in traci.trafficlight.getControlledLanes(affected_tls))
                        spike[affected_tls] = max(50.0, queue_now * 5.0)
                        self.coordinator.update_pheromones(spike)

    def _clear_old_incidents(self, duration: int = 120):
        to_clear = [l for l, t in self._blocked_lanes.items()
                    if self.step_count - t >= duration]
        for lane in to_clear:
            traci.lane.setAllowed(lane, ["passenger", "truck", "bus"])
            del self._blocked_lanes[lane]

    # ── observation ───────────────────────────────────────────────────────────
    def _apply_actions(self, actions: dict):
        current_time = traci.simulation.getTime()
        for tls_id, action in actions.items():
            if action == 1:
                # enforce minimum phase hold — prevent rapid oscillation
                elapsed = float(traci.trafficlight.getNextSwitch(tls_id) - current_time)
                phase_hold = getattr(self, '_phase_hold', 5.0)
                if elapsed > -phase_hold:   # only switch if held long enough
                    current  = traci.trafficlight.getPhase(tls_id)
                    n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
                    traci.trafficlight.setPhase(tls_id, (current + 1) % n_phases)


    def _get_obs(self, tls_id: str) -> np.ndarray:
        lanes  = traci.trafficlight.getControlledLanes(tls_id)
        queues = {d: 0.0 for d in ("N", "S", "E", "W")}

        for lane in lanes:
            queues[self._lane_direction(lane)] += traci.lane.getLastStepHaltingNumber(lane)

        phase_id     = int(traci.trafficlight.getPhase(tls_id))
        phase_onehot = np.zeros(MAX_PHASES, dtype=np.float32)
        phase_onehot[min(phase_id, MAX_PHASES - 1)] = 1.0  # clip if a junction exceeds MAX_PHASES

        phase_elapsed = float(traci.trafficlight.getNextSwitch(tls_id)
                              - traci.simulation.getTime())

        # pheromone signal replaces simple nbr_max_queue
        if self.coordinator:
            nbr_signal = self.coordinator.neighbour_pheromone(
                tls_id, self._neighbours.get(tls_id, []), hops=int(round(self._nbr_radius))
            )
        else:
            nbr_signal = self._fallback_nbr_queue(tls_id)

        raw = np.concatenate([
            np.array([queues["N"], queues["S"], queues["E"], queues["W"]], dtype=np.float32),
            phase_onehot,
            np.array([
                phase_elapsed, nbr_signal,
                self.coordinator.incoming_signal(tls_id) if self.coordinator else 0.0,
                self.coordinator._downstream_space.get(tls_id, 1.0) if self.coordinator else 1.0,
                self.coordinator.incident_signal(tls_id) if self.coordinator else 0.0,
            ], dtype=np.float32),
        ])

# running normalisation (Welford's algorithm) — true accumulated
        # variance across the episode, not a single-step snapshot
        self._obs_n += 1
        if self._obs_mean is None:
            self._obs_mean = raw.copy()
            self._obs_m2   = np.zeros_like(raw)
        else:
            delta          = raw - self._obs_mean
            self._obs_mean += delta / self._obs_n
            delta2         = raw - self._obs_mean
            self._obs_m2   += delta * delta2
        variance = self._obs_m2 / max(self._obs_n - 1, 1)
        self._obs_std = np.sqrt(variance)

        # normalise only continuous features; one-hot phase block stays untouched (already 0/1)
        raw_norm  = raw.copy()
        phase_end = 4 + MAX_PHASES
        continuous = [0, 1, 2, 3] + list(range(phase_end, phase_end + 3))
        raw_norm[continuous] = (raw[continuous] - self._obs_mean[continuous]) / (self._obs_std[continuous] + 1e-8)

        self._obs_buffer[tls_id].append(raw_norm)
        return np.stack(self._obs_buffer[tls_id])


    def _get_all_obs(self) -> dict:
        return {tls: self._get_obs(tls) for tls in self.tls_ids}

    def _get_reward(self, tls_id: str) -> float:
        lanes      = traci.trafficlight.getControlledLanes(tls_id)
        total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        throughput = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
        base = (-total_wait / max(len(lanes), 1)) + (0.3 * throughput)
        if self.coordinator and self.coordinator.incident_signal(tls_id) > 0:
            base += 0.5 * throughput
        return base

    def _get_all_rewards(self) -> dict:
        return {tls: self._get_reward(tls) for tls in self.tls_ids}

    def _coordinate_green_wave(self):
         
        pass 

    def _update_max_pressure(self):
        if not self.coordinator:
            return
        downstream_space = {}
        for tls in self.tls_ids:
            nbrs = self._neighbours.get(tls, [])
            if nbrs:
                # downstream space = inverse of downstream queue pressure
                nbr_queues = [
                    sum(traci.lane.getLastStepHaltingNumber(l)
                        for l in traci.trafficlight.getControlledLanes(n))
                    for n in nbrs
                ]
                downstream_space[tls] = 1.0 / (1.0 + np.mean(nbr_queues))
            else:
                downstream_space[tls] = 1.0
        # store in coordinator for potential obs use
        self.coordinator._downstream_space = downstream_space

    # ── neighbour map (built once on reset) ───────────────────────────────────

    def _build_neighbour_map(self) -> dict:
        # Pre-compute 1-hop neighbours for each TLS. Called 1 time/reset 
        nbr_map = {tls: [] for tls in self.tls_ids}
        tls_lane_sets = {
            tls: set(traci.trafficlight.getControlledLanes(tls))
            for tls in self.tls_ids
        }
        for tls in self.tls_ids:
            for link_group in traci.trafficlight.getControlledLinks(tls):
                for link in link_group:
                    edge = link[1].rsplit("_", 1)[0]
                    for other in self.tls_ids:
                        if other != tls:
                            if any(l.startswith(edge) for l in tls_lane_sets[other]):
                                if other not in nbr_map[tls]:
                                    nbr_map[tls].append(other)
        return nbr_map

    def _fallback_nbr_queue(self, tls_id: str) -> float:
        """Used if no coordinator is attached."""
        max_q = 0.0
        for nbr in self._neighbours.get(tls_id, []):
            for lane in traci.trafficlight.getControlledLanes(nbr):
                max_q = max(max_q, traci.lane.getLastStepHaltingNumber(lane))
        return max_q

    @staticmethod
    def _lane_direction(lane_id: str) -> str:
        l = lane_id.lower()
        if any(k in l for k in ("north", "_n_", "nb")): return "N"
        if any(k in l for k in ("south", "_s_", "sb")): return "S"
        if any(k in l for k in ("east",  "_e_", "eb")): return "E"
        return "W"