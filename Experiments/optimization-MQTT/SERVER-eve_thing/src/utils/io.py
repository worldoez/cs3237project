from pathlib import Path
import json, joblib, torch

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_joblib(obj, path):
    ensure_dir(Path(path).parent)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)

def save_torch(model, path):
    ensure_dir(Path(path).parent)
    torch.save(model.state_dict(), path)

def load_torch(model, path, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model