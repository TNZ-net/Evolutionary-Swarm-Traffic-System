import os
import json
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ── agent checkpoints ─────────────────────────────────────────────────────────

def save_agent(agent, path: str):
    """Save SB3 PPO agent weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agent.save(path)

def load_agent(path: str, fallback_agent=None):
    from stable_baselines3 import PPO
    zip_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(zip_path):
        print(f"  ⚠ no trained weights at {zip_path} — using untrained/random agent")
        return fallback_agent
    return PPO.load(path)

# ── run logging ───────────────────────────────────────────────────────────────
class RunLogger:
    def __init__(self, log_dir: str = "checkpoints", use_wandb: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, f"run_{timestamp()}.csv")
        self._headers_written = False
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                #wandb.init(project="sumo-swarm", reinit=True)
                wandb.init(project="sumo-swarm", entity="2webwandpyf-n", reinit=True)
                self._wandb = wandb
            except Exception:
                print("wandb not available — logging to CSV only.")

    def log(self, metrics: dict):
        if not self._headers_written:
            with open(self._path, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self._headers_written = True

        with open(self._path, "a") as f:
            f.write(",".join(str(v) for v in metrics.values()) + "\n")

        if self._wandb:
            self._wandb.log(metrics)

    def close(self):
        if self._wandb:
            self._wandb.finish()