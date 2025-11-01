
# HW2 — Deep Q-Network (DQN)

This repo provides a PyTorch implementation of DQN for **CartPole-v1** and **MsPacman-v0** with:
- Replay Buffer
- Target Network
- ε-greedy policy with linear decay
- Optional Huber loss & LR scheduler
- Required plots (max-Q vs episodes, rewards vs episodes with moving average, 500-episode rollout histogram + mean/std)

## Setup

```bash
# Create virtual environment
python -m venv .venv_gpu
source .venv_gpu/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Train

### CartPole
```bash
python src/train_cartpole.py --episodes 800 --gamma 0.95
```

### MsPacman
```bash
python src/train_mspacman.py --episodes 5000 --gamma 0.99
```

Outputs (plots, logs) are saved under `outputs/` with timestamped run folders.

## Generate 500-episode Rollout Histogram (after training)
Each training script automatically runs a 500-episode evaluation using the best checkpoint and saves the histogram & stats to the same run folder.

