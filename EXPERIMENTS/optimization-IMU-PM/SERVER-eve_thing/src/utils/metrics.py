import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report
    }