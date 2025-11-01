import argparse, json
import numpy as np
import torch
from pathlib import Path
import gymnasium as gym

from utils import set_seed, device_from_str, now_ts, ensure_dir
from replay_buffer import ReplayBuffer
from models import QNetworkMLP
from dqn import DQNAgent, DQNConfig
from plots import plot_max_q, plot_rewards, plot_hist

def make_env(env_id: str):
    return gym.make(env_id)

def train(args):
    env = make_env("CartPole-v1")
    test_env = make_env("CartPole-v1")

    set_seed(args.seed)
    device = device_from_str(args.device)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNetworkMLP(obs_dim, n_actions, hidden=args.hidden)
    qt = QNetworkMLP(obs_dim, n_actions, hidden=args.hidden)

    cfg = DQNConfig(
        gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
        train_start=args.train_start, target_update=args.target_update,
        eps_start=args.eps_start, eps_end=args.eps_end, eps_decay_episodes=args.eps_decay_episodes,
        huber=args.huber, device=device,
    )
    agent = DQNAgent(q, qt, cfg, n_actions)

    rb = ReplayBuffer(args.buffer_capacity, state_shape=(obs_dim,), state_dtype=np.float32)

    run_dir = Path("outputs") / f"cartpole_{now_ts()}"
    ensure_dir(run_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rewards, max_qs = [], []
    best_reward = -1e9

    for ep in range(1, args.episodes + 1):
        s, info = env.reset(seed=args.seed)
        ep_reward, done, steps, out = 0.0, False, 0, None

        while not done and steps < args.max_steps:
            a, eps = agent.act(np.asarray(s, dtype=np.float32), ep)
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            rb.push(np.asarray(s, dtype=np.float32), a, float(r), np.asarray(s2, dtype=np.float32), float(done))
            s = s2
            ep_reward += float(r)
            steps += 1
            out = agent.update(rb, ep)

        rewards.append(ep_reward)
        max_qs.append(out["max_q"] if out is not None else (max_qs[-1] if max_qs else 0.0))

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.q.state_dict(), run_dir / "best.pt")

        if ep % 10 == 0 or ep == 1:
            print(f"[EP {ep:4d}] reward={ep_reward:.1f} eps={eps:.3f} len={steps} best={best_reward:.1f}")

    # Plots
    plot_max_q(max_qs, run_dir / "max_q_vs_episodes.png")
    plot_rewards(rewards, run_dir / "rewards_vs_episodes.png", window=100)

    # 500-episode evaluation (greedy)
    eval_rewards = []
    agent.q.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    for _ in range(500):
        s, info = test_env.reset()
        done, ep_r, steps = False, 0.0, 0
        while not done and steps < args.max_steps:
            s_t = torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                a = int(torch.argmax(agent.q(s_t), dim=1).item())
            s, r, terminated, truncated, info = test_env.step(a)
            done = terminated or truncated
            ep_r += float(r)
            steps += 1
        eval_rewards.append(ep_r)

    eval_rewards = np.array(eval_rewards, dtype=np.float32)
    plot_hist(eval_rewards, run_dir / "eval_hist_500.png", bins=30, title="CartPole 500-episode Reward Histogram")
    with open(run_dir / "eval_stats.json", "w") as f:
        json.dump({"mean": float(eval_rewards.mean()), "std": float(eval_rewards.std())}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--buffer_capacity", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_start", type=int, default=1000)
    ap.add_argument("--target_update", type=int, default=1000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=400)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--huber", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train(args)
