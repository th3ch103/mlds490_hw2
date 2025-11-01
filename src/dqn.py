from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    train_start: int = 1000
    target_update: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 500
    huber: bool = True
    device: str = "cpu"


class DQNAgent:
    """
    DQN Agent with:
      - Epsilon-greedy policy
      - Target network
      - Huber/MSE loss
      - Robust input shaping for both vector (CartPole) and image (MsPacman) obs
    """
    def __init__(self, qnet, qtarget, cfg: DQNConfig, n_actions: int):
        self.q = qnet
        self.qt = qtarget
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()

        self.n_actions = n_actions
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.q.to(self.device)
        self.qt.to(self.device)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.global_step = 0

    def epsilon(self, ep: int) -> float:
        """Linear decay of epsilon over episodes."""
        t = min(1.0, ep / max(1, self.cfg.eps_decay_episodes))
        return self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs, ep: int):
        """
        obs: can be
          - vector (D,) for CartPole
          - image (C,H,W) for MsPacman (already transposed to CHW by caller)
        Ensures a batch dimension for both cases.
        """
        eps = self.epsilon(ep)
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions), eps

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            # (D,) -> (1, D)
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.ndim == 3:
            # (C,H,W) -> (1, C, H, W)
            obs_t = obs_t.unsqueeze(0)

        qvals = self.q(obs_t)
        return int(torch.argmax(qvals, dim=1).item()), eps

    def update(self, replay, ep: int):
        """One DQN gradient step (if replay is warm)."""
        if len(replay) < self.cfg.train_start:
            return None

        s, a, r, s2, d = replay.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)

        # If replay stored images as HWC, permute to CHW for CNNs.
        # (B, H, W, C) -> (B, C, H, W)
        if s_t.ndim == 4 and s_t.shape[1] != 1 and s_t.shape[-1] in (1,3,4):
            s_t  = s_t.permute(0, 3, 1, 2).contiguous()
            s2_t = s2_t.permute(0, 3, 1, 2).contiguous()

        # Safety: if somehow a single sample sneaks in as 3-D, add batch dim.
        if s_t.ndim == 3:
            s_t  = s_t.unsqueeze(0)
            s2_t = s2_t.unsqueeze(0)

        # Q(s,a)
        q_vals = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

        # target = r + (1-d) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.qt(s2_t).max(1)[0]
            target = r_t + (1.0 - d_t) * self.cfg.gamma * next_q

        loss = F.smooth_l1_loss(q_vals, target) if self.cfg.huber else F.mse_loss(q_vals, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update == 0:
            self.qt.load_state_dict(self.q.state_dict())

        with torch.no_grad():
            max_q = self.q(s_t).max(1)[0].mean().item()

        return {"loss": float(loss.item()), "max_q": max_q}
