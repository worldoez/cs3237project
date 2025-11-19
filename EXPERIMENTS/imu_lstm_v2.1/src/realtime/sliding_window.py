from collections import deque
import numpy as np

class SlidingWindow:
    def __init__(self, length: int, n_feat: int):
        self.length = int(length)
        self.n_feat = int(n_feat)
        self.buf = deque(maxlen=int(length))

    def push(self, x):
        arr = np.asarray(x, dtype=np.float32).reshape(self.n_feat)
        self.buf.append(arr)

    def ready(self):
        return len(self.buf) == self.length

    def array(self):
        if not self.ready():
            raise ValueError("window not ready")
        return np.vstack(self.buf)