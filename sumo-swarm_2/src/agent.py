import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from config import OBS_DIM, SEQ_LEN, GRU_HIDDEN, GRU_LAYERS,TOTAL_TIMESTEPS 


# ── GRU feature extractor (plugs into SB3) ────────────────────────────────────
class GRUExtractor(BaseFeaturesExtractor):
    """
    Accepts obs of shape (SEQ_LEN, OBS_DIM).
    Outputs h_t of shape (GRU_HIDDEN,) — the temporal context vector.
    """

    def __init__(self, observation_space: gym.spaces.Box, eff_seq_len: int = None):
        super().__init__(observation_space, features_dim=GRU_HIDDEN)
        self.eff_seq_len = eff_seq_len or observation_space.shape[0]
        self.gru = nn.GRU(
            input_size=OBS_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            batch_first=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, SEQ_LEN, OBS_DIM) — slice to this agent's evolved effective window
        obs = obs[:, -self.eff_seq_len:, :]
        _, h_n = self.gru(obs)   # h_n: (num_layers, batch, hidden)
        return h_n[-1]           # (batch, GRU_HIDDEN)

# ── Value network (learned V(s) replacing scalar proxy) ──────────────────────
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = None   # set after init via setup_optimizer()

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.net(h_t).squeeze(-1)

    def setup_optimizer(self, lr: float = 1e-3):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def update(self, h_t: torch.Tensor, target: float):
        """One-step TD update toward target V(s)."""
        if self.optimizer is None:
            self.setup_optimizer()
        pred  = self.forward(h_t)
        loss  = nn.functional.mse_loss(pred, torch.tensor([target]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred.item()

def make_value_net() -> ValueNetwork:
    return ValueNetwork()

# ── observation space helper ──────────────────────────────────────────────────

def make_obs_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(SEQ_LEN, OBS_DIM),
        dtype=np.float32,
    )


def make_action_space() -> gym.spaces.Discrete:
    return gym.spaces.Discrete(2)   # 0 = keep phase, 1 = next phase


# ── agent factory ─────────────────────────────────────────────────────────────

def make_agent(chromosome: dict = None) -> PPO:
    from config import LEARNING_RATE, GAMMA, N_STEPS, BATCH_SIZE, N_EPOCHS

    n_steps = int(chromosome.get("buf_size", N_STEPS)) if chromosome else N_STEPS
    n_steps = max(64, (n_steps // 64) * 64)
    lr    = chromosome.get("lr",    LEARNING_RATE) if chromosome else LEARNING_RATE
    gamma = chromosome.get("gamma", GAMMA)         if chromosome else GAMMA
    ent_coef = chromosome.get("ent_coef", 0.01) if chromosome else 0.01
    epsilon  = chromosome.get("epsilon", 0.10)  if chromosome else 0.10
    ent_coef = ent_coef + (epsilon * 0.05)   # folds the previously-unused epsilon gene into PPO's real exploration knob

    # SB3 needs a dummy env to infer spaces — we pass specs directly
    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = make_obs_space()
            self.action_space      = make_action_space()
        def reset(self, **kw):  return self.observation_space.sample(), {}
        def step(self, a):      return self.observation_space.sample(), 0.0, False, False, {}

    seq_len = int(chromosome.get("seq_len", SEQ_LEN)) if chromosome else SEQ_LEN
    policy_kwargs = dict(
        features_extractor_class=GRUExtractor,
        features_extractor_kwargs={"eff_seq_len": seq_len},
        net_arch=[64, 64],      # MLP layers after GRU output
    )

    agent = PPO(
        policy="MlpPolicy",
        env=_DummyEnv(),
        learning_rate=lr,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    agent._ent_coef_start = ent_coef
    return agent


def predict(agent: PPO, obs: np.ndarray) -> int:
    """Single-step inference. obs shape: (SEQ_LEN, OBS_DIM)."""
    action, _ = agent.predict(obs[np.newaxis], deterministic=True)
    return int(action[0])

def predict_with_hidden(agent: PPO, obs: np.ndarray):
    """Returns (action, h_t) — exposes GRU hidden state."""
    import torch
    obs_tensor, _ = agent.policy.obs_to_tensor(obs[np.newaxis])
    with torch.no_grad():
        features = agent.policy.extract_features(
            obs_tensor, agent.policy.pi_features_extractor
        )
    action, _ = agent.predict(obs[np.newaxis], deterministic=True)
    return int(action[0]), features.squeeze(0)


def init_for_training(agent: PPO) -> PPO:
    from config import TOTAL_TIMESTEPS
    agent._setup_learn(TOTAL_TIMESTEPS, reset_num_timesteps=True)
    return agent


def predict_train(agent: PPO, obs: np.ndarray): 
    import torch
    obs_tensor, _ = agent.policy.obs_to_tensor(obs[np.newaxis])
    with torch.no_grad():
        features = agent.policy.extract_features(
            obs_tensor, agent.policy.pi_features_extractor
        )
        actions, values, log_probs = agent.policy.forward(obs_tensor)
    return int(actions[0].item()), values, log_probs, features.squeeze(0)


def store_and_maybe_train(agent: PPO, obs: np.ndarray, action: int, reward: float,
                           episode_start: bool, value, log_prob, next_obs: np.ndarray):
     
    import numpy as np, torch
    buf = agent.rollout_buffer
    buf.add(
        obs[np.newaxis], np.array([action]), np.array([reward]),
        np.array([episode_start]), value, log_prob,
    )
    agent.num_timesteps += 1

    if not buf.full:
        return False

    with torch.no_grad():
        next_tensor, _ = agent.policy.obs_to_tensor(next_obs[np.newaxis])
        _, next_value, _ = agent.policy.forward(next_tensor)
    buf.compute_returns_and_advantage(last_values=next_value, dones=np.array([False]))
    import config as _cfg
    progress = min(agent.num_timesteps / _cfg.EXPECTED_CANDIDATE_STEPS, 1.0)
    if agent.num_timesteps >= getattr(agent, "_ent_coef_boost_until", 0):
        agent.ent_coef = agent._ent_coef_start * (1.0 - 0.9 * progress)
    agent.train()
    buf.reset()
    return True