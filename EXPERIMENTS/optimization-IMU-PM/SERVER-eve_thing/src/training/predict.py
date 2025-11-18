import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

# make src importable
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import load_joblib, load_json
from data.preprocessing import load_raw_dir
from data.windowing import window_by_session, FEATURE_ORDER
from models.lstm_v2 import LSTMIMU

def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="imu_data_*.csv file or folder containing them")
    ap.add_argument("--artifacts", required=True, help="folder with lstm_v2_best.pth, scaler.joblib, label_encoder.joblib, config.json")
    ap.add_argument("--window", type=int, default=0, help="0=use effective_window from config.json")
    ap.add_argument("--stride", type=int, default=0, help="0=use effective_stride from config.json")
    ap.add_argument("--out", default="predictions.csv")
    return ap.parse_args()

def main():
    a = args()
    art = Path(a.artifacts)
    scaler = load_joblib(art / "scaler.joblib")
    le = load_joblib(art / "label_encoder.joblib")
    cfg = load_json(art / "config.json")["config"]
    eff_window = a.window if a.window > 0 else cfg.get("effective_window", cfg.get("window", 150))
    eff_stride = a.stride if a.stride > 0 else cfg.get("effective_stride", cfg.get("stride", 25))

    # load data (file or dir)
    src = Path(a.input)
    if src.is_dir():
        df, _ = load_raw_dir(str(src))
    else:
        df = pd.read_csv(src)
        if "session" not in df.columns:
            df = df.copy()
            df["session"] = src.stem
        df["action_label"] = df["action_label"].str.strip().str.lower()

    X, y_str, meta = window_by_session(df, window=eff_window, stride=eff_stride)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMIMU(input_size=6, hidden_size=cfg["hidden"], num_layers=cfg["layers"],
                    dropout=cfg["dropout"], num_classes=len(le.classes_), bidirectional=cfg["bidirectional"]).to(device)
    model.load_state_dict(torch.load(art / "lstm_v2_best.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        idx = probs.argmax(axis=1)
        labels = le.inverse_transform(idx)

    # write CSV with probabilities per class (ordered as le.classes_)
    out = Path(a.out)
    prob_cols = [f"prob_{c}" for c in le.classes_]
    rows = []
    for i, m in enumerate(meta):
        row = {
            "session": m["session"],
            "start": m["start"],
            "true_label": y_str[i],
            "pred_label": labels[i],
            "confidence": float(probs[i, idx[i]]),
        }
        for j, c in enumerate(le.classes_):
            row[prob_cols[j]] = float(probs[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out.resolve()} with {len(rows)} rows (window={eff_window}, stride={eff_stride})")

if __name__ == "__main__":
    main()