import json, glob, os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning

ROOT = os.path.join(os.path.dirname(__file__), "../../models_artifacts")
ROOT = os.path.abspath(ROOT)
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1) Training/validation curves (if history.json exists)
hist_file = os.path.join(ROOT, "history.json")
if os.path.exists(hist_file):
    with open(hist_file, "r") as f:
        H = json.load(f)
    # H expected keys: 'train_loss','val_loss','train_acc','val_acc' lists of length epochs
    epochs = range(1, len(H.get("train_loss", [])) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs, H.get("train_loss", []), label="train_loss")
    plt.plot(epochs, H.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epochs, H.get("train_acc", []), label="train_acc")
    plt.plot(epochs, H.get("val_acc", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "acc_curve.png"))
    plt.close()
else:
    print("No history.json found, skipping train/val curves.")

# 2) Read predictions CSV (try main predictions then logs)
pred_files = []
p_main = os.path.join(ROOT, "predictions.csv")
if os.path.exists(p_main):
    pred_files.append(p_main)
# add logs
# pred_files += sorted(glob.glob(os.path.join(ROOT, "logs", "predictions_log_*.csv")))

if not pred_files:
    print("No prediction CSVs found in", ROOT)
else:
    # concatenate and pick useful columns
    dfs = []
    for f in pred_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print("skip", f, e)
    if not dfs:
        raise SystemExit("No readable prediction CSVs")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    # Expect columns similar to: timestamp, device_id, label, command, confidence, prob_jump, prob_left, prob_right, prob_straight, raw_gx...
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    # choose true label column: 'label' or 'true'
    # if 'label' not in df.columns:
    #     raise SystemExit("predictions CSV missing 'label' column")
    # y_true = df['label'].astype(str)
    # # choose predicted label column: 'command' or 'pred'
    # pred_col = 'command' if 'command' in df.columns else 'pred'
    # if pred_col not in df.columns:
    #     raise SystemExit("predictions CSV missing 'command' column")
    # y_pred = df[pred_col].astype(str)

    # Determine true label column (support multiple formats)
    if 'label' in df.columns:
        true_col = 'label'
    elif 'y_true' in df.columns:
        true_col = 'y_true'
    elif 'true' in df.columns:
        true_col = 'true'
    else:
        raise SystemExit("predictions CSV missing true-label column (label / y_true / true)")

    # Determine predicted label column
    if 'command' in df.columns:
        pred_col = 'command'
    elif 'pred' in df.columns:
        pred_col = 'pred'
    elif 'y_pred' in df.columns:
        pred_col = 'y_pred'
    else:
        raise SystemExit("predictions CSV missing predicted-column (command / pred / y_pred)")

    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    # Silence sklearn undefined metric warnings (we still produce report)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    labels_present = sorted(list(set(y_true.unique()).union(set(y_pred.unique()))))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_present, yticklabels=labels_present, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Classification report (zero_division=0 to avoid warnings in CSV)
    report = classification_report(y_true, y_pred, labels=labels_present, output_dict=True, zero_division=0)
    rpt_df = pd.DataFrame(report).transpose()
    rpt_df.to_csv(os.path.join(PLOTS_DIR, "classification_report.csv"))

    # Confidence distribution: attempt to find prob_ columns
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols:
        def get_prob(row):
            pred = str(row[pred_col]).lower()
            key = "prob_" + pred
            return float(row.get(key, np.nan))
        df['pred_prob'] = df.apply(get_prob, axis=1)
        df['correct'] = (y_true == y_pred)
        plt.figure(figsize=(8,4))
        sns.histplot(data=df, x='pred_prob', hue='correct', bins=30, kde=False, stat="density", common_norm=False)
        plt.xlabel("Predicted class probability")
        plt.title("Confidence distribution: correct vs incorrect")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "confidence_distribution.png"))
        plt.close()
    else:
        print("No prob_ columns found; skipping confidence distribution.")
    
    # Save a sample of errors for manual inspection using available columns
    # pick a sensible set of columns if present
    candidate_cols = ['timestamp','device_id', true_col, pred_col, 'confidence','raw_gx','raw_gy','raw_gz','raw_ax','raw_ay','raw_az']
    existing_cols = [c for c in candidate_cols if c in df.columns]
    errors = df[df[pred_col] != df[true_col]]
    if not errors.empty:
        errors[existing_cols].head(200).to_csv(os.path.join(PLOTS_DIR, "sample_errors.csv"), index=False)
    else:
        # write an empty file indicating no errors
        pd.DataFrame(columns=existing_cols).to_csv(os.path.join(PLOTS_DIR, "sample_errors.csv"), index=False)

print("Plots & reports written to", PLOTS_DIR)