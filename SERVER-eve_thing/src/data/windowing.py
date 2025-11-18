import numpy as np
import pandas as pd
from typing import Tuple, Dict

FEATURE_ORDER = ["gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"]

def window_by_session(
    df: pd.DataFrame, window: int = 150, stride: int = 25
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create sliding windows within each (session, label) block.
    Returns:
      X: (N, window, 6)
      y: (N,)
      meta: list of dicts with session, start_idx, label
    """
    X_list, y_list, meta = [], [], []
    for (session, label), g in df.groupby(["session","action_label"]):
        g = g.reset_index(drop=True)
        feats = g[FEATURE_ORDER].values.astype(np.float32)
        n = len(g)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            seg = feats[start:start+window]
            X_list.append(seg)
            y_list.append(label)
            meta.append({"session": session, "start": int(start), "label": label})
    if not X_list:
        raise ValueError("No windows produced. Reduce window size or check data length.")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, meta