import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import OBS_DIM, SEQ_LEN, ROUTE_FILES
from env import SumoSwarmEnv


# ── Student network ───────────────────────────────────────────────────────────
class StudentMLP(nn.Module):
    """
    Tiny MLP replacing GRU-PPO at inference time.
    Input:  flattened obs window → (SEQ_LEN * OBS_DIM,)
    Output: action logits        → (2,)
    """
    def __init__(self, input_dim: int = SEQ_LEN * OBS_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Teacher data collection ───────────────────────────────────────────────────

def collect_teacher_data(teacher, env: SumoSwarmEnv, n_episodes: int) -> tuple:
    """
    Rolls out the trained PPO teacher, records (flat_obs, action_logits) pairs.
    Matches agent.py predict() signature exactly:
        agent.predict(obs[np.newaxis], deterministic=True)
    where obs shape is (SEQ_LEN, OBS_DIM).
    """
    X, Y = [], []

    for ep in range(n_episodes):
        obs_dict = env.reset()
        done = False

        while not done:
            actions = {}
            for aid, obs in obs_dict.items():
                # obs shape: (SEQ_LEN, OBS_DIM) — matches agent.py exactly
                obs_batch = obs[np.newaxis]   # → (1, SEQ_LEN, OBS_DIM)

                # get action logits from PPO policy internals
                obs_tensor, _ = teacher.policy.obs_to_tensor(obs_batch)
                with torch.no_grad():
                    # GRUExtractor forward → features → policy/value heads
                    features = teacher.policy.extract_features(
                        obs_tensor, teacher.policy.pi_features_extractor
                    )
                    latent_pi = teacher.policy.mlp_extractor.forward_actor(features)
                    logits = teacher.policy.action_net(latent_pi).cpu().numpy()[0]

                X.append(obs.flatten())   # (SEQ_LEN*OBS_DIM,)
                Y.append(logits)          # (2,)

                # action for stepping env — same as predict() in agent.py
                action, _ = teacher.predict(obs_batch, deterministic=True)
                actions[aid] = int(action[0])

            obs_dict, _, done, _ = env.step(actions)

        print(f"  Episode {ep+1}/{n_episodes} — {len(X)} samples")

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ── Distillation training ─────────────────────────────────────────────────────

def distil(X: np.ndarray, Y: np.ndarray, epochs: int = 50, lr: float = 1e-3) -> StudentMLP:
    """KL divergence loss: student log-probs vs teacher softmax probs."""
    student   = StudentMLP()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    kl_loss   = nn.KLDivLoss(reduction="batchmean")

    X_t           = torch.tensor(X)
    teacher_probs = torch.softmax(torch.tensor(Y), dim=-1)

    dataset = torch.utils.data.TensorDataset(X_t, teacher_probs)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = kl_loss(torch.log_softmax(student(xb), dim=-1), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total/len(loader):.5f}")

    return student


# ── ONNX export ───────────────────────────────────────────────────────────────
def export_onnx(student: StudentMLP, path: str):
    student.eval()
    dummy = torch.zeros(1, SEQ_LEN * OBS_DIM)
    torch.onnx.export(
        student, dummy, path,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=11,
    )
    print(f"ONNX exported → {path}")


def print_ncnn_steps(onnx_path: str):
    print(f"""
── NCNN conversion ───────────────────────────────────────────────
1. Download ncnn tools: https://github.com/Tencent/ncnn/releases
2. onnx2ncnn {onnx_path} student.param student.bin
3. Deploy student.param + student.bin on edge device per junction
4. Inference: flatten obs (SEQ_LEN*OBS_DIM,) → ncnn → argmax(2) = action
──────────────────────────────────────────────────────────────────
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from stable_baselines3 import PPO
    from agent import GRUExtractor   # must be importable for PPO.load to rebuild policy

    print(f"Loading teacher from {args.teacher} ...")
    teacher = PPO.load(args.teacher, custom_objects={
        "features_extractor_class": GRUExtractor
    })

    print("Collecting rollouts from teacher...")
    env = SumoSwarmEnv(route_file=ROUTE_FILES["peak"])
    X, Y = collect_teacher_data(teacher, env, n_episodes=args.episodes)
    env.close()
    print(f"Dataset ready: {X.shape[0]} samples, input dim {X.shape[1]}")

    print("Training student MLP...")
    student = distil(X, Y, epochs=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    pt_path   = os.path.join(args.out_dir, "student.pt")
    onnx_path = os.path.join(args.out_dir, "student.onnx")

    torch.save(student.state_dict(), pt_path)
    print(f"Weights saved → {pt_path}")

    export_onnx(student, onnx_path)
    print_ncnn_steps(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher",  type=str, required=True,
                        help="Path to trained PPO .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--out_dir",  type=str, default="checkpoints")
    main(parser.parse_args())