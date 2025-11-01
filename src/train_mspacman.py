import argparse
import json
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from utils import set_seed, device_from_str, now_ts, ensure_dir
from replay_buffer import ReplayBuffer
from models import QNetworkCNN
from dqn import DQNAgent, DQNConfig
from plots import plot_max_q, plot_rewards, plot_hist

# Assignment-specified color (sum of RGB) for contrast improvement
MSPACMAN_COLOR = 210 + 164 + 74


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """
    Assignment preprocessing:
      1) Crop/downsample: obs[1:176:2, ::2]
      2) Grayscale by sum over RGB channels
      3) Zero out pixels equal to MSPACMAN_COLOR
      4) Normalize to int8 in [-128, 127]
      5) Return (88, 80, 1)
    """
    img = obs[1:176:2, ::2]        # crop and downsample
    img = img.sum(axis=2)          # grayscale (sum over channels)
    img[img == MSPACMAN_COLOR] = 0 # improve contrast
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)


def make_env(env_id: str):
    """Create Gymnasium ALE env that returns RGB frames."""
    return gym.make(env_id, obs_type="rgb")


def train(args: argparse.Namespace) -> None:
    """
    Train a DQN agent on MsPacman using Gymnasium:
      - Replay buffer of preprocessed int8 frames (H,W,C) = (88,80,1)
      - Agent takes CHW tensors; we transpose at call sites
      - Target network, Huber loss, epsilon-greedy
      - Auto-save best model by episode reward
      - Plots + 500-episode greedy evaluation
    """
    ENV_ID = "ALE/MsPacman-v5"

    env = make_env(ENV_ID)
    test_env = make_env(ENV_ID)

    set_seed(args.seed)
    device = device_from_str(args.device)

    n_actions = env.action_space.n  # expected 9

    # Build CNN Q-networks
    q = QNetworkCNN(n_actions=n_actions)
    qt = QNetworkCNN(n_actions=n_actions)

    # Optional: confirm inferred flat_dim (should be 2688 with (1,64,7,6))
    print("[QNetworkCNN] inferred flat_dim:", q.head[1].in_features)

    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        train_start=args.train_start,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        huber=args.huber,
        device=device,
    )
    agent = DQNAgent(q, qt, cfg, n_actions)

    # Replay buffer stores frames as int8 (HWC). We'll permute to CHW inside agent.update().
    rb = ReplayBuffer(args.buffer_capacity, state_shape=(88, 80, 1), state_dtype=np.int8)

    run_dir = Path("outputs") / f"mspacman_{now_ts()}"
    ensure_dir(run_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rewards, max_qs = [], []
    best_reward = -1e9

    # ----------------------------
    # Training loop
    # ----------------------------
    for ep in range(1, args.episodes + 1):
        s_raw, _ = env.reset(seed=args.seed)
        s = preprocess_observation(s_raw)

        ep_reward = 0.0
        done = False
        steps = 0
        out = None

        while not done and steps < args.max_steps:
            # Agent expects CHW; s is HWC -> transpose to CHW
            a, eps = agent.act(np.asarray(s).transpose(2, 0, 1), ep)

            s2_raw, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            s2 = preprocess_observation(s2_raw)

            # Reward clipping (stabilizes Atari training); keep original reward for logging
            rb.push(s, a, float(np.clip(r, -1.0, 1.0)), s2, float(done))
            s = s2
            ep_reward += float(r)
            steps += 1

            out = agent.update(rb, ep)

        rewards.append(ep_reward)
        max_qs.append(out["max_q"] if out is not None else (max_qs[-1] if max_qs else 0.0))

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.q.state_dict(), run_dir / "best.pt")

        if ep % 20 == 0 or ep == 1:
            print(f"[EP {ep:5d}] reward={ep_reward:.1f} eps={eps:.3f} len={steps} best={best_reward:.1f}")

    # ----------------------------
    # Training plots
    # ----------------------------
    plot_max_q(max_qs, run_dir / "max_q_vs_episodes.png")
    plot_rewards(rewards, run_dir / "rewards_vs_episodes.png", window=100)

    # ----------------------------
    # 500-episode greedy evaluation
    # ----------------------------
    eval_rewards = []
    agent.q.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    for _ in range(500):
        s_raw, _ = test_env.reset()
        s = preprocess_observation(s_raw)

        done = False
        ep_r = 0.0
        steps = 0

        while not done and steps < args.max_steps:
            s_t = torch.as_tensor(
                s.transpose(2, 0, 1), dtype=torch.float32, device=agent.device
            ).unsqueeze(0)  # (1,C,H,W)
            with torch.no_grad():
                a = int(torch.argmax(agent.q(s_t), dim=1).item())

            s2_raw, r, terminated, truncated, _ = test_env.step(a)
            done = bool(terminated or truncated)
            s = preprocess_observation(s2_raw)
            ep_r += float(r)
            steps += 1

        eval_rewards.append(ep_r)

    eval_rewards = np.array(eval_rewards, dtype=np.float32)
    plot_hist(eval_rewards, run_dir / "eval_hist_500.png", bins=30, title="MsPacman 500-episode Reward Histogram")
    with open(run_dir / "eval_stats.json", "w") as f:
        json.dump({"mean": float(eval_rewards.mean()), "std": float(eval_rewards.std())}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Core training length
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=10000)

    # DQN hyperparameters
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer_capacity", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--train_start", type=int, default=5000)
    ap.add_argument("--target_update", type=int, default=5000)

    # Epsilon schedule (linear in episodes)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay_episodes", type=int, default=3000)

    # Misc
    ap.add_argument("--huber", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    train(args)

