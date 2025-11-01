import torch
import torch.nn as nn


# -------------------------
# MLP for CartPole (vector)
# -------------------------
class QNetworkMLP(nn.Module):
    """
    Simple MLP for low-dimensional observations (e.g., CartPole).
    Input:  (B, obs_dim)
    Output: (B, n_actions)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# CNN for MsPacman (images)
# -------------------------
class QNetworkCNN(nn.Module):
    """
    DQN-style CNN for Atari frames.

    Assumptions:
      - Input is CHANNEL-FIRST (B, 1, 88, 80), preprocessed per assignment.
      - Values approximately in [-128, 127] (int8). We scale to ~[-1, 1] in forward().

    Architecture:
      Conv(1,32,8,4) -> ReLU
      Conv(32,64,4,2) -> ReLU
      Conv(64,64,3,1) -> ReLU
      Flatten -> Linear(flat_dim -> 512) -> ReLU -> Linear(512 -> n_actions)

    flat_dim is **inferred dynamically** with a dummy forward pass
    to avoid hard-coded sizes (e.g., 2688).
    """
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 88, 80, dtype=torch.float32)  # (B=1,C=1,H=88,W=80)
            out = self.features(dummy)                              # -> (1, C', H', W') expected (1,64,7,6)
            flat_dim = out.view(1, -1).size(1)                      # 64*7*6 = 2688

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 88, 80), dtype int8/float32
        - Cast to float32 if needed
        - Scale by 1/128 to map roughly to [-1, 1]
        """
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 128.0
        x = self.features(x)
        return self.head(x)
