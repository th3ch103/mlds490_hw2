
# HW2 — Deep Q-Network (DQN)

This repo provides a clean, minimal PyTorch implementation of DQN for **CartPole-v1** and **MsPacman-v0** with:
- Replay Buffer
- Target Network
- ε-greedy policy with linear decay
- Optional Huber loss & LR scheduler
- Required plots (max-Q vs episodes, rewards vs episodes with moving average, 500-episode rollout histogram + mean/std)

> Tested with Python 3.10+. If your environment differs (e.g., deepdish cluster), adapt the `requirements.txt` versions accordingly.

---

## Setup

```bash
# (Recommended) Create venv/conda then install:
pip install -r requirements.txt
# For ALE ROMs (Ms. Pac-Man), you may need:
# python -m AutoROM --accept-license
```

> If your gym uses the newer Gymnasium API, set `--gymnasium` on the training script.

---

## Train

### CartPole
```bash
python src/train_cartpole.py --episodes 800 --gamma 0.95 --seed 42
```

### MsPacman
```bash
python src/train_mspacman.py --episodes 5000 --gamma 0.99 --seed 42
```

Outputs (plots, logs) are saved under `outputs/` with timestamped run folders.

---

## Generate 500-episode Rollout Histogram (after training)
Each training script automatically runs a 500-episode evaluation using the best checkpoint and saves the histogram & stats to the same run folder.

---

## Notes
- MsPacman pre-processing strictly follows the assignment:
  - crop/downsample → grayscale → remove Pac-Man color → normalize to [-128, 127] and reshape to (88, 80, 1).
- To speed up training, you can reduce replay warmup, batch size, or network width, but performance may degrade.
- All hyperparameters can be changed via CLI flags.
