from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLS = ["action_id","timestamp","gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z","action_label"]
CLASS_NAMES = ["left","right","straight","jump"]

def load_raw_dir(raw_dir: str):
    raw_dir = Path(raw_dir)
    files = sorted([p for p in raw_dir.glob("imu_data_*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No imu_data_*.csv files found in {raw_dir}")
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} missing columns: {missing}")
        df = df.copy()
        df["session"] = p.stem  # prevent windowing across files
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Normalize labels (lowercase)
    data["action_label"] = data["action_label"].str.strip().str.lower()

    # Filter to expected classes only
    data = data[data["action_label"].isin(CLASS_NAMES)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows with expected labels {left,right,straight,jump}")

    # Ensure dtype
    for c in ["gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"])

    # Basic stats for sanity
    stats = data.groupby("action_label")[["accel_x","accel_y","accel_z"]].mean().to_dict()
    # Optional: check sampling interval roughness (median delta ~20ms)
    data = data.sort_values(["session","timestamp"]).reset_index(drop=True)
    return data, stats