import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make "src" importable when run as a script
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.seed import set_seed
from utils.io import ensure_dir, save_json, save_joblib, save_torch
from utils.metrics import evaluate_predictions
from data.preprocessing import load_raw_dir
from data.windowing import window_by_session
from data.datasets import IMUWindowDataset
from models.lstm_v2 import LSTMIMU

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Path to raw CSV dir (imu_data2)")
    ap.add_argument("--out_dir", required=True, help="Where to save artifacts")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load
    df, stats = load_raw_dir(args.raw_dir)

    # Auto-adjust window if needed
    min_len = int(df.groupby(["session","action_label"]).size().min())
    eff_window = int(args.window)
    eff_stride = int(args.stride)
    if min_len < eff_window:
        eff_window = max(16, min_len)
        eff_stride = max(1, eff_window // 6)
        print(f"[INFO] Requested window {args.window} too large for dataset (min segment {min_len}). "
              f"Using window={eff_window}, stride={eff_stride}.")

    # Window
    X, y_str, meta = window_by_session(df, window=eff_window, stride=eff_stride)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Split
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    val_frac = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_frac, random_state=args.seed, stratify=y_tmp)

    # Scale per-channel
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)
    def apply_scaler(arr):
        s = scaler.transform(arr.reshape(-1, arr.shape[-1]))
        return s.reshape(arr.shape)
    X_train = apply_scaler(X_train)
    X_val = apply_scaler(X_val)
    X_test = apply_scaler(X_test)

    # Datasets
    ds_train = IMUWindowDataset(X_train, y_train)
    ds_val   = IMUWindowDataset(X_val, y_val)
    ds_test  = IMUWindowDataset(X_test, y_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch, shuffle=False)
    dl_test  = DataLoader(ds_test, batch_size=args.batch, shuffle=False)

    # Model
    model = LSTMIMU(
        input_size=6,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        num_classes=len(le.classes_),
        bidirectional=args.bidirectional
    ).to(device)

    # Class weights
    class_counts = np.bincount(y_train, minlength=len(le.classes_))
    weights = (class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # History containers
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc, best_path = -1.0, out_dir / "lstm_v2_best.pth"
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum, n_sum = 0.0, 0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * xb.size(0)
            n_sum += xb.size(0)
            pbar.set_postfix(loss=loss_sum/n_sum)

        # Compute train loss/acc
        train_loss = (loss_sum / n_sum) if n_sum > 0 else 0.0
        train_acc = evaluate_loop(model, dl_train, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Compute val loss
        val_acc = evaluate_loop(model, dl_val, device)
        val_loss = evaluate_loss(model, dl_val, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
        print(f"Val acc: {val_acc:.4f} (best {best_val_acc:.4f})  train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f}")


    # Test with best
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_pred = infer_all(model, dl_test, device)
    from utils.metrics import evaluate_predictions
    metrics = evaluate_predictions(y_test.tolist(), y_pred.tolist(), labels=list(range(len(le.classes_))))
    metrics["label_classes"] = le.classes_.tolist()

    # Save artifacts
    save_joblib(scaler, out_dir / "scaler.joblib")
    save_joblib(le, out_dir / "label_encoder.joblib")
    save_json({
        "config": {
            **vars(args),
            "effective_window": eff_window,
            "effective_stride": eff_stride,
        },
        "stats": stats,
        "best_val_acc": best_val_acc
    }, out_dir / "config.json")
    pd.DataFrame({"y_true": le.inverse_transform(y_test), "y_pred": le.inverse_transform(y_pred)}).to_csv(out_dir / "predictions.csv", index=False)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }
    with open(out_dir / "history.json", "w") as fh:
        json.dump(history, fh)

    print("Done. Artifacts saved to", out_dir)

def evaluate_loop(model, dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
    return correct / max(total, 1)

def evaluate_loss(model, dl, device, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += float(loss) * xb.size(0)
            n += xb.size(0)
    return loss_sum / max(n, 1)

def infer_all(model, dl, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            outs.append(pred.cpu())
    return torch.cat(outs, dim=0)

if __name__ == "__main__":
    main()