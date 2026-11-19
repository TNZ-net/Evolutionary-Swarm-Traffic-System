"""
config.py - tunable variables


    NETWORK_PROFILE=grid3x3_2tl HYBRID_SELECTION=performance GENOME_MODE=per_agent python train.py --evolution pso --generations 25
    NETWORK_PROFILE=grid3x3 HYBRID_SELECTION=performance GENOME_MODE=per_agent python train.py --evolution hybrid --generations 25

"""
import os
import shutil

def _find_sumo_home() -> str:
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"] 
    binary = shutil.which("sumo") or shutil.which("sumo.exe")
    if binary:
        return str(os.path.dirname(os.path.dirname(binary)))
    raise EnvironmentError(
        "SUMO_HOME not found. Set it as an environment variable or add sumo to PATH."
    )

# ── SUMO ──────────────────────────────────────────────────────────────────────
SUMO_HOME = _find_sumo_home()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NETWORK_PROFILE = os.environ.get("NETWORK_PROFILE", "grid3x3")   # toggle: "full" | "grid2x2" | "grid4x4" | "grid3x3"

_NETWORK_PATHS = {
    "full": {
        "net": os.path.join(PROJECT_ROOT, "network", "SwarmNetwork.net.xml"),
        "routes": {
            "peak":    "network/demand_peak.rou.xml",
            "offpeak": "network/demand_offpeak.rou.xml",
            "weekend": "network/demand_weekend.rou.xml",
        },
    },
    "grid2x2": {
        "net": os.path.join(PROJECT_ROOT, "simple_network", "grid2x2", "net.xml"),
        "routes": {
            "peak":    "simple_network/grid2x2/demand_peak.rou.xml",
            "offpeak": "simple_network/grid2x2/demand_offpeak.rou.xml",
            "weekend": "simple_network/grid2x2/demand_weekend.rou.xml",
        },
    },
    "grid3x3": {
        "net": os.path.join(PROJECT_ROOT, "simple_network", "grid3x3", "net.xml"),
        "routes": {
            "peak":    "simple_network/grid3x3/demand_peak.rou.xml",
            "offpeak": "simple_network/grid3x3/demand_offpeak.rou.xml",
            "weekend": "simple_network/grid3x3/demand_weekend.rou.xml",
        },
    },
    "grid3x3_2tl": {
        "net": os.path.join(PROJECT_ROOT, "simple_network", "grid3x3", "net_2tl.net.xml"),
        "routes": {
            "peak":    "simple_network/grid3x3/demand_peak.rou.xml",
            "offpeak": "simple_network/grid3x3/demand_offpeak.rou.xml",
            "weekend": "simple_network/grid3x3/demand_weekend.rou.xml",
        },
    },
    "grid3x3_6tl": {
    "net": os.path.join(PROJECT_ROOT, "simple_network", "grid3x3_6tl", "net.xml"),
    "routes": {
        "peak":    "simple_network/grid3x3_6tl/demand_peak.rou.xml",
        "offpeak": "simple_network/grid3x3_6tl/demand_offpeak.rou.xml",
        "weekend": "simple_network/grid3x3_6tl/demand_weekend.rou.xml",
    },
},
}

NET_FILE    = _NETWORK_PATHS[NETWORK_PROFILE]["net"]
ROUTE_FILES = _NETWORK_PATHS[NETWORK_PROFILE]["routes"]

SIM_STEP_LENGTH  = 1 
USE_GUI          = False

# Test
GA_POPULATION = 8
GA_GENERATIONS = 25
EPISODE_STEPS  = 900
N_STEPS          = 256

# ── OBSERVATION ───────────────────────────────────────────────────────────────
# OBS_DIM          = 8 # [q_N, q_S, q_E, q_W, phase_id, phase_elapsed, nbr_pheromone, incoming]
MAX_PHASES       = 8  # upper bound on traffic-light phases across the network; phase is one-hot encoded to this width
#OBS_DIM          = 7 + MAX_PHASES  # [q_N, q_S, q_E, q_W, phase_onehot(MAX_PHASES), phase_elapsed, nbr_pheromone, incoming]
OBS_DIM          = 9 + MAX_PHASES  # [q_N, q_S, q_E, q_W, phase_onehot(MAX_PHASES), phase_elapsed, nbr_pheromone, incoming, downstream_space, incident_signal]
SEQ_LEN          = 15 

# ── GRU ENCODER ───────────────────────────────────────────────────────────────
GRU_HIDDEN       = 64
GRU_LAYERS       = 1

# ── RL (PPO) ──────────────────────────────────────────────────────────────────
LEARNING_RATE    = 3e-4
GAMMA            = 0.95 # PPO and Bellman bootstrap in coordinator
BATCH_SIZE       = 64
N_EPOCHS         = 10
TOTAL_TIMESTEPS  = 500_000
EXPECTED_CANDIDATE_STEPS = 2700 

# ── REWARD ────────────────────────────────────────────────────────────────────
ALPHA            = 0.25

# ── GA ────────────────────────────────────────────────────────────────────────
GA_MUTATION_RATE = 0.08
GA_ELITE_N       = 2
HEAL_THRESHOLD   = -500.0
HEAL_STREAK      = 3

GENE_BOUNDS = {
    "lr":          [1e-5, 1e-3],
    "gamma":       [0.90, 0.99],
    "epsilon":     [0.05, 0.30],
    "nbr_radius":  [0,    2   ],
    "alpha":       [0.10, 0.50],
    "phase_hold":  [5,    60  ],
    "buf_size":    [128,  1024],
    "seq_len":     [5,    15  ],   # ≤ SEQ_LEN — sliced, never exceeds the real window
    "ent_coef":    [0.001, 0.05], 
}

# ── CHECKPOINTING ─────────────────────────────────────────────────────────────
#NETWORK_PROFILE=grid3x3 CHECKPOINT_SUBFOLDER=incident_v2 HYBRID_SELECTION=proximity GENOME_MODE=per_agent python train.py --evolution hybrid --generations 25
CHECKPOINT_SUBFOLDER = os.environ.get("CHECKPOINT_SUBFOLDER", "")
CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "checkpoints", NETWORK_PROFILE, CHECKPOINT_SUBFOLDER
)

# ── TRAINING ──────────────────────────────────────────────────────────────────
SCENARIO_WEIGHTS = {"peak": 0.7, "offpeak": 0.15, "weekend": 0.15}
POPULATION_RESTART_EVERY = 8   # inject random chromosomes every N generations (used when RESTART_MODE == "fixed")
RESTART_MODE = os.environ.get("RESTART_MODE", "fixed")  # "fixed" (default, current behaviour) | "fitness" (trigger on stagnation instead of a fixed clock)
RESTART_STAGNATION_GENS = 3    # "fitness" mode only: consecutive generations with no improvement before injecting random chromosomes
MAX_PRESSURE_WEIGHT = 0.3       # downstream space weight in phase selection
SAVE_EVERY_N_GEN         = 5

#── TOGGLEBLES ─────────────────────────────────────────────────────────────
#HYBRID_SELECTION = "proximity"  # toggle: "proximity" or "performance"
#HYBRID_SELECTION = "performance"
HYBRID_SELECTION = os.environ.get("HYBRID_SELECTION", "proximity")
GENOME_MODE = os.environ.get("GENOME_MODE", "per_agent")

#── GENOME MODE (SETSI toggle) ─────────────────────────────────────
GENE_SHARE_RATE   = 0.15       # per_agent only: prob. a junction adopts a fitter neighbour's genes
GENE_SHARE_EVERY  = 3          # per_agent only: generations between neighbour gene-sharing events
GENE_SHARE_TARGET = os.environ.get("GENE_SHARE_TARGET", "last")  # "last" (default, current behaviour) | "weakest" (replace lowest-fitness slot instead)