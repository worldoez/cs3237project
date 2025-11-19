import argparse, sys, json
from pathlib import Path
import numpy as np, serial, torch

# Make "src" importable when run as a script
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import load_joblib
from models.lstm_v2 import LSTMIMU
from realtime.sliding_window import SlidingWindow

MOTOR_COMMANDS = {
    "left": "LEFT",
    "right": "RIGHT",
    "straight": "STRAIGHT",
    "jump": "JUMP"
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--window", type=int, default=0, help="0=use effective window from config.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="Min prob to emit label")
    return ap.parse_args()

def main():
    args = parse_args()
    art = Path(args.artifacts)
    scaler = load_joblib(art / "scaler.joblib")
    le = load_joblib(art / "label_encoder.joblib")
    cfgj = json.load(open(art / "config.json"))
    cfg = cfgj["config"]
    eff_window = args.window if args.window > 0 else cfg.get("effective_window", cfg.get("window", 150))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMIMU(input_size=6, hidden_size=cfg["hidden"], num_layers=cfg["layers"],
                    dropout=cfg["dropout"], num_classes=len(le.classes_), bidirectional=cfg["bidirectional"]).to(device)
    state = torch.load(art / "lstm_v2_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    win = SlidingWindow(eff_window, n_feat=6)

    print("Ready. Expecting lines: gx,gy,gz,ax,ay,az", file=sys.stderr)

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            parts = [float(x) for x in line.split(",")]
            if len(parts) != 6:
                continue
            # Order matches training: gx,gy,gz,ax,ay,az
            win.push(parts)
            if not win.ready():
                continue
            arr = win.array()[None, ...]  # (1,T,C)
            arr = scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
            xb = torch.tensor(arr, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(probs.argmax())
                label = le.inverse_transform([idx])[0]
                conf = float(probs[idx])
            # Only emit if above threshold; otherwise emit NONE
            if conf >= args.threshold:
                cmd = MOTOR_COMMANDS.get(label, label.upper())
                print(f"{cmd},{conf:.3f}")
            else:
                print(f"NONE,{conf:.3f}")
        except Exception:
            continue

if __name__ == "__main__":
    main()