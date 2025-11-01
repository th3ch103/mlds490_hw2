from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape, state_dtype=np.uint8):
        self.capacity = capacity
        self.mem = deque(maxlen=capacity)
        self.state_shape = state_shape
        self.state_dtype = state_dtype

    def push(self, s, a, r, s2, d):
        self.mem.append((s, a, r, s2, d))

    def __len__(self):
        return len(self.mem)

    def sample(self, batch_size: int):
        batch = random.sample(self.mem, batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = np.stack(s).astype(self.state_dtype)
        s2 = np.stack(s2).astype(self.state_dtype)
        a  = np.array(a, dtype=np.int64)
        r  = np.array(r, dtype=np.float32)
        d  = np.array(d, dtype=np.float32)
        return s, a, r, s2, d
