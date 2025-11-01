import os, time, json, math, random
import numpy as np
import torch
from dataclasses import dataclass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_from_str(s: str):
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s

def moving_avg(x, k=100):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0: return x
    w = min(k, len(x))
    c = np.convolve(x, np.ones(w)/w, mode='valid')
    pad = np.empty(len(x) - len(c))
    pad[:] = np.nan
    return np.concatenate([pad, c])

def now_ts():
    import datetime as dt
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
