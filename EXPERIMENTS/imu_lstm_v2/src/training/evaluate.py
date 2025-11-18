import argparse, sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

# Make "src" importable when run as a script
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import load_joblib, load_json
from data.preprocessing import load_raw_dir
from data.windowing import window_by_session
from data.datasets import IMUWindowDataset
from models.lstm_v2 import LSTMIMU

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--window", type=int, default=0, help="0=use effective window from config.json")
    ap.add_argument("--stride", type=int, default=0, help="0=use effective stride from config.json")
    return ap.parse_args()

def main():
    args = get_args()
    art = Path(args.artifacts)
    scaler = load_joblib(art / "scaler.joblib")
    le = load_joblib(art / "label_encoder.joblib")
    cfg = load_json(art / "config.json")["config"]
    eff_window = args.window if args.window > 0 else cfg.get("effective_window", cfg.get("window", 150))
    eff_stride = args.stride if args.stride > 0 else cfg.get("effective_stride", cfg.get("stride", 25))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _ = load_raw_dir(args.raw_dir)
    X, y_str, _ = window_by_session(df, window=eff_window, stride=eff_stride)
    y = le.transform(y_str)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    ds = IMUWindowDataset(X, y)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    model = LSTMIMU(input_size=6, hidden_size=cfg["hidden"], num_layers=cfg["layers"],
                    dropout=cfg["dropout"], num_classes=len(le.classes_), bidirectional=cfg["bidirectional"]).to(device)
    model.load_state_dict(torch.load(art / "lstm_v2_best.pth", map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
    print(f"Accuracy: {correct/max(total,1):.4f} (window={eff_window}, stride={eff_stride})")

if __name__ == "__main__":
    main()