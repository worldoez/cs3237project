# import paho.mqtt.client as mqtt
# import time
# import argparse, sys, json, os, atexit, csv
# import numpy as np, torch
# from pathlib import Path
# from typing import Dict, Any
# from collections import deque
# from datetime import datetime
# from threading import Lock

# # Make "src" importable
# SRC_DIR = Path(__file__).resolve().parents[1]
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# from utils.io import load_joblib, load_json
# from models.lstm_v2 import LSTMIMU
# from realtime.sliding_window import SlidingWindow
# from realtime.motion_control import ensure_control_state, update_control

# # MQTT Configuration
# BROKER_IP = "10.180.115.152"  # Update this to your broker IP
# BROKER_PORT = 1883
# TOPIC_IMU_IN = "/imu"
# TOPIC_IMU_OUT = "/imu_out"
# TOPIC_SLEEP = "/eve_sleep"

# YAW_IDX = {"gx": 0, "gy": 1, "gz": 2}

# state = {
#     "scaler": None,
#     "le": None,
#     "cfg": None,
#     "model": None,
#     "device": None,
#     "eff_window": None,
#     "artifacts": None,
#     "threshold": 0.6,
#     "per_class_threshold": {},
#     "debounce_k": 2,
#     "min_acc_for_jump": 12.0,
#     "debounce_map": {"default": 3, "straight": 2},
#     "threshold_map": {"default": 0.6, "straight": 0.55, "left": 0.7, "right": 0.7},
#     "quiet_gyro": 0.06,
#     "quiet_ms": 250,
#     "last_quiet": {},
#     "turn_fast_thr": 0.90,
#     "turn_min_gyro": 0.10,
#     "ema_probs": {},
#     "ema_alpha": 0.6,
#     "yaw_assist": True,
#     "yaw_axis": "gx",
#     "yaw_sign": 1,
#     "yaw_min": 0.12,
#     "yaw_window": 12,
#     "yaw_prior_k": 2.0,
#     "yaw_prior_min": 0.08,
#     "yaw_prior_max": 0.35,
#     "probs_s": {},
#     "jump_hold_ms": 140,
#     "jump_lock_until": {},
#     "ctl": {},
#     "max_speed": 1.0,
#     "speed_alpha": 0.25,
#     "yaw_alpha": 0.30,
#     "max_yaw_rate_dps": 180.0,
#     "turn_mix": 0.6,
#     "acc_to_speed_gain": 0.08,
#     "acc_bias_alpha": 0.01,
#     "speed_floor": 0.0,
#     "auto_calib": 0,
#     "auto_calib_done": {},
#     "turn_guard_ms": 150.0,
#     "last_turn": {},
#     "windows": {},
#     "latest": {},
#     "recent": {},
#     "rot": {},
#     "g_ref": None,
#     "hist": {},
#     "current_cmd": {},
#     "classes": [],
#     "log_rows": [],
#     "log_dir": None,
#     "auto_calib_on_first_data": True,
#     "calib_done_devices": set(),
#     "jump_reset_delay_ms": 1500,
#     "jump_detected_at": {},
# }

# state_lock = Lock()

# AXIS_MAP = {"idx": [0, 1, 2, 3, 4, 5], "sign": [1, 1, 1, 1, 1, 1]}


# def load_axis_map(artifacts: Path):
#     path = artifacts / "axis_map.json"
#     if path.exists():
#         try:
#             cfg = json.load(open(path))
#             idx = cfg.get("idx", AXIS_MAP["idx"])
#             sign = cfg.get("sign", AXIS_MAP["sign"])
#             if len(idx) == 6 and len(sign) == 6:
#                 return {"idx": [int(i) for i in idx], "sign": [int(s) for s in sign]}
#         except Exception:
#             pass
#     return AXIS_MAP


# def apply_map(parts):
#     out = [0.0] * 6
#     for i in range(6):
#         out[i] = AXIS_MAP["sign"][i] * parts[AXIS_MAP["idx"][i]]
#     return out


# def load_gravity_ref(artifacts: Path):
#     path = artifacts / "gravity_ref.json"
#     if path.exists():
#         try:
#             j = json.load(open(path))
#             g = np.array(j.get("g_ref", [0, 9.81, 0]), dtype=float)
#             if g.shape == (3,):
#                 return g
#         except Exception:
#             pass
#     return np.array([0.0, 9.81, 0.0], dtype=float)


# def normalize(v):
#     n = np.linalg.norm(v)
#     return v / n if n > 1e-9 else v


# def rot_from_vectors(a, b):
#     a_u, b_u = normalize(a), normalize(b)
#     v = np.cross(a_u, b_u)
#     c = np.dot(a_u, b_u)
#     if np.linalg.norm(v) < 1e-9:
#         return np.eye(3) if c > 0 else -np.eye(3)
#     s = np.linalg.norm(v)
#     vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
#     return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


# def apply_rotation(parts, device_id: str):
#     R = state["rot"].get(device_id, None)
#     if R is None:
#         return parts
#     g = np.array(parts[0:3], dtype=float)
#     a = np.array(parts[3:6], dtype=float)
#     g_r = (R @ g).tolist()
#     a_r = (R @ a).tolist()
#     return g_r + a_r


# def _rot_path(device_id: str) -> Path:
#     d = Path(state["artifacts"]) / "calib"
#     d.mkdir(parents=True, exist_ok=True)
#     return d / f"rot_{device_id}.json"


# def save_rot(device_id: str, R):
#     p = _rot_path(device_id)
#     with open(p, "w") as f:
#         json.dump({"R": (R.tolist() if hasattr(R, "tolist") else R)}, f)


# def load_rot(device_id: str):
#     p = _rot_path(device_id)
#     if p.exists():
#         try:
#             j = json.load(open(p))
#             return np.array(j["R"], dtype=float)
#         except Exception:
#             pass
#     return None


# def init_model(artifacts: Path, threshold: float, window_override: int = 0):
#     global AXIS_MAP
#     AXIS_MAP = load_axis_map(artifacts)
#     scaler = load_joblib(artifacts / "scaler.joblib")
#     le = load_joblib(artifacts / "label_encoder.joblib")
#     cfg = load_json(artifacts / "config.json")["config"]
#     eff_window = (
#         window_override
#         if window_override > 0
#         else cfg.get("effective_window", cfg.get("window", 150))
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMIMU(
#         input_size=6,
#         hidden_size=cfg["hidden"],
#         num_layers=cfg["layers"],
#         dropout=cfg["dropout"],
#         num_classes=len(le.classes_),
#         bidirectional=cfg["bidirectional"],
#     ).to(device)
#     model.load_state_dict(
#         torch.load(artifacts / "lstm_v2_best.pth", map_location=device)
#     )
#     model.eval()
#     state.update(
#         dict(
#             scaler=scaler,
#             le=le,
#             cfg=cfg,
#             model=model,
#             device=device,
#             eff_window=eff_window,
#             threshold=threshold,
#             artifacts=str(artifacts),
#         )
#     )
#     state["g_ref"] = load_gravity_ref(artifacts)
#     log_dir = Path(artifacts) / "logs"
#     log_dir.mkdir(parents=True, exist_ok=True)
#     state["log_dir"] = str(log_dir)
#     print("[INFO] Axis map idx=", AXIS_MAP["idx"], "sign=", AXIS_MAP["sign"])
#     print("[INFO] g_ref =", state["g_ref"].tolist())


# def ensure_window(device_id: str):
#     if device_id not in state["windows"]:
#         state["windows"][device_id] = SlidingWindow(state["eff_window"], n_feat=6)
#     if device_id not in state["recent"]:
#         state["recent"][device_id] = deque(maxlen=300)
#     if device_id not in state["rot"]:
#         R = load_rot(device_id)
#         state["rot"][device_id] = R if R is not None else np.eye(3)
#     if device_id not in state["hist"]:
#         state["hist"][device_id] = deque(maxlen=8)
#     if device_id not in state["last_quiet"]:
#         state["last_quiet"][device_id] = 0.0
#     ensure_control_state(state, device_id)


# def _gate_jump_and_decide(probs: np.ndarray, arr_raw: np.ndarray):
#     classes = list(state["le"].classes_)
#     idx = int(np.argmax(probs))
#     if classes[idx].lower() == "jump":
#         acc = arr_raw[:, 3:6]
#         g_hat = normalize(state["g_ref"])
#         a_proj_abs = np.abs(acc @ g_hat)
#         a_p90 = float(np.percentile(a_proj_abs, 90.0))
#         if a_p90 < float(state["min_acc_for_jump"]):
#             probs[idx] = 0.0
#             idx = int(np.argmax(probs))
#     return classes[idx], float(probs[idx]), probs, idx


# def _apply_yaw_prior(device_id: str, probs: np.ndarray, arr_raw: np.ndarray) -> np.ndarray:
#     classes = list(state["le"].classes_)
#     try:
#         li = classes.index("left")
#         ri = classes.index("right")
#     except ValueError:
#         return probs
#     yaw_idx = YAW_IDX.get(state.get("yaw_axis", "gx"), 0)
#     n = max(3, min(int(state.get("yaw_window", 12)), arr_raw.shape[0]))
#     yaw = arr_raw[-n:, yaw_idx].mean() * float(state.get("yaw_sign", 1))
#     a = float(state.get("yaw_prior_min", 0.08))
#     b = float(state.get("yaw_prior_max", 0.35))
#     mag = max(0.0, min(1.0, (abs(yaw) - a) / max(1e-6, (b - a))))
#     if mag <= 0.0:
#         return probs
#     k = float(state.get("yaw_prior_k", 2.0))
#     logits = np.log(np.clip(probs, 1e-8, 1.0))
#     if yaw > 0:
#         logits[ri] += k * mag
#     elif yaw < 0:
#         logits[li] += k * mag
#     e = np.exp(logits - logits.max())
#     return e / e.sum()


# def _ema_update(device_id: str, probs: np.ndarray) -> np.ndarray:
#     alpha = float(state.get("ema_alpha", 0.6))
#     prev = state["ema_probs"].get(device_id, None)
#     out = probs if prev is None else (alpha * probs + (1.0 - alpha) * prev)
#     s = float(np.sum(out))
#     if s > 1e-9:
#         out = out / s
#     state["ema_probs"][device_id] = out
#     state["probs_s"][device_id] = out
#     return out


# def _debounced_command(device_id: str, label: str, confidence: float, now_ts: float, arr_raw: np.ndarray) -> str:
#     g = arr_raw[:, :3]
#     g_mag = float(np.sqrt((g**2).sum(axis=1)).mean())
#     if g_mag < float(state["quiet_gyro"]):
#         state["last_quiet"][device_id] = now_ts

#     th_map = state["threshold_map"]
#     th_def = float(th_map.get("default", state["threshold"]))
#     th_straight = float(th_map.get("straight", th_def))
#     th_label = float(th_map.get(label.lower(), th_def))

#     jump_detected_at = state.get("jump_detected_at", {}).get(device_id, 0.0)
#     reset_delay_ms = float(state.get("jump_reset_delay_ms", 1500))
    
#     if jump_detected_at > 0:
#         elapsed_ms = (now_ts - jump_detected_at) * 1000.0
#         if elapsed_ms >= reset_delay_ms:
#             print(f"[INFO] Jump reset period over for {device_id}, clearing state")
#             _reset_device_state(device_id, keep_calibration=True)
#             return "NONE"
#         else:
#             return "JUMP"

#     if label.lower() == "straight" and confidence >= th_straight:
#         state["current_cmd"][device_id] = "STRAIGHT"
#         state["hist"][device_id].clear()
#         state.get("last_turn", {}).pop(device_id, None)
#         return "STRAIGHT"

#     classes = list(state["le"].classes_)
#     try:
#         j_idx = classes.index("jump")
#     except ValueError:
#         j_idx = None

#     lock_until = state["jump_lock_until"].get(device_id, 0.0)
#     if now_ts < float(lock_until):
#         state["current_cmd"][device_id] = "JUMP"
#         return "JUMP"

#     if j_idx is not None:
#         probs_s = state["probs_s"].get(device_id)
#         p_jump = float(probs_s[j_idx]) if probs_s is not None else 0.0
#         th_jump = float(th_map.get("jump", th_def))
#         acc = arr_raw[:, 3:6]
#         g_hat = normalize(state["g_ref"])
#         a_p90 = float(np.percentile(np.abs(acc @ g_hat), 90.0))
#         if (p_jump >= float(state["turn_fast_thr"])) or (
#             p_jump >= th_jump and a_p90 >= float(state["min_acc_for_jump"])
#         ):
#             state["current_cmd"][device_id] = "JUMP"
#             state["jump_lock_until"][device_id] = (
#                 now_ts + float(state["jump_hold_ms"]) / 1000.0
#             )
#             state["jump_detected_at"][device_id] = now_ts
#             print(f"[INFO] JUMP detected for {device_id} at {now_ts}")
#             return "JUMP"

#     if label.lower() != "straight":
#         if confidence >= float(state["turn_fast_thr"]) and g_mag >= float(state["turn_min_gyro"]):
#             guard_ms = float(state.get("turn_guard_ms", 150.0))
#             last = state.get("last_turn", {}).get(device_id, {"dir": None, "ts": 0.0})
#             opp = "LEFT" if label.upper() == "RIGHT" else "RIGHT"
#             if (
#                 last
#                 and last.get("dir") == opp
#                 and (now_ts - float(last.get("ts", 0.0))) * 1000.0 < guard_ms
#             ):
#                 pass
#             else:
#                 state["current_cmd"][device_id] = label.upper()
#                 state["hist"][device_id].clear()
#                 state.setdefault("last_turn", {})[device_id] = {
#                     "dir": label.upper(),
#                     "ts": now_ts,
#                 }
#                 return label.upper()

#     k_map = state["debounce_map"]
#     k_def = int(k_map.get("default", state["debounce_k"]))
#     k = int(k_map.get(label.lower(), k_def))
#     hist = state["hist"][device_id]
#     hist.append(label)

#     if confidence < th_label:
#         return state["current_cmd"].get(device_id, "NONE")

#     last_k = list(hist)[-k:]
#     if len(last_k) == k and all(l == label for l in last_k):
#         state["current_cmd"][device_id] = label.upper()
#         if label.lower() != "straight":
#             state.setdefault("last_turn", {})[device_id] = {
#                 "dir": label.upper(),
#                 "ts": now_ts,
#             }
#         else:
#             state.get("last_turn", {}).pop(device_id, None)
#     return state["current_cmd"].get(device_id, "NONE")


# def _reset_device_state(device_id: str, keep_calibration: bool = True):
#     saved_rot = None
#     saved_calib_status = False
    
#     if keep_calibration:
#         saved_rot = state["rot"].get(device_id)
#         saved_calib_status = device_id in state.get("calib_done_devices", set())
    
#     state["windows"].pop(device_id, None)
#     state["recent"].pop(device_id, None)
#     state["hist"].pop(device_id, None)
#     state["last_quiet"].pop(device_id, None)
#     state["latest"].pop(device_id, None)
#     state["current_cmd"].pop(device_id, None)
#     state["jump_lock_until"].pop(device_id, None)
#     state["jump_detected_at"].pop(device_id, None)
#     state["ema_probs"].pop(device_id, None)
#     state["probs_s"].pop(device_id, None)
#     state.get("last_turn", {}).pop(device_id, None)
#     state.get("ctl", {}).pop(device_id, None)
#     state["auto_calib_done"].pop(device_id, None)
    
#     if keep_calibration and saved_rot is not None:
#         state["rot"][device_id] = saved_rot
#         if saved_calib_status:
#             state.setdefault("calib_done_devices", set()).add(device_id)
    
#     ensure_window(device_id)
#     print(f"[INFO] Reset state for {device_id} (calibration preserved: {keep_calibration})")


# def calibrate_gravity(device_id: str, client):
#     ensure_window(device_id)
#     win = state["windows"][device_id]
#     if not win.ready():
#         return {"ok": False, "error": "window not ready"}
#     arr = win.array()
#     g_live = arr[:, 3:6].mean(axis=0)
#     g_ref = state["g_ref"]
#     R = rot_from_vectors(g_live, g_ref)
#     state["rot"][device_id] = R
#     save_rot(device_id, R)
#     return {
#         "ok": True,
#         "g_live": g_live.tolist(),
#         "g_ref": g_ref.tolist(),
#         "R": R.tolist(),
#     }


# def auto_yaw(device_id: str):
#     win = state["windows"].get(device_id)
#     if not win or not win.ready():
#         return {"ok": False, "error": "window not ready"}
#     arr = win.array()
#     stds = arr[:, 0:3].std(axis=0)
#     axes = ["gx", "gy", "gz"]
#     axis = axes[int(np.argmax(stds))]
#     state["yaw_axis"] = axis
#     return {
#         "ok": True,
#         "axis": axis,
#         "sign": state["yaw_sign"],
#         "std": {axes[i]: float(stds[i]) for i in range(3)},
#     }


# def process_imu_data(payload: Dict[str, Any], client):
#     """Process incoming IMU data and publish results"""
#     with state_lock:
#         device_id = str(payload.get("device_id", "imu01"))
#         samples = payload.get("samples", [])
        
#         ensure_window(device_id)
#         win = state["windows"][device_id]
        
#         for s in samples:
#             parts = [
#                 float(s["gx"]), float(s["gy"]), float(s["gz"]),
#                 float(s["ax"]), float(s["ay"]), float(s["az"]),
#             ]
#             parts = apply_map(parts)
#             parts = apply_rotation(parts, device_id)
#             ts = int(s.get("ts", time.time() * 1000))
#             latest_sample = {
#                 "ts": ts,
#                 "gx": parts[0], "gy": parts[1], "gz": parts[2],
#                 "ax": parts[3], "ay": parts[4], "az": parts[5],
#             }
#             state["recent"][device_id].append(latest_sample)
#             win.push(parts)
        
#         # Check calibration
#         if state.get("auto_calib", 0) and device_id not in state.get("calib_done_devices", set()):
#             if win.ready():
#                 print(f"[INFO] Performing auto-calibration for {device_id}")
#                 calibrate_gravity(device_id, client)
#                 time.sleep(0.1)
#                 auto_yaw(device_id)
#                 state.setdefault("calib_done_devices", set()).add(device_id)
                
#                 response = {
#                     "ok": True,
#                     "status": "calibrating",
#                     "message": "Calibration complete"
#                 }
#                 client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
#                 return
#             else:
#                 response = {
#                     "ok": True,
#                     "status": "calibrating",
#                     "message": "Collecting calibration data...",
#                     "progress": f"{win.count}/{state['eff_window']}"
#                 }
#                 client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
#                 return
        
#         # Normal processing
#         pred = None
#         if win.ready():
#             arr_raw = win.array()
#             arr = arr_raw[None, ...]
#             arr_s = (
#                 state["scaler"]
#                 .transform(arr.reshape(-1, arr.shape[-1]))
#                 .reshape(arr.shape)
#             )
#             xb = torch.tensor(arr_s, dtype=torch.float32).to(state["device"])
#             with torch.no_grad():
#                 logits = state["model"](xb)
#                 probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

#             now_ts = time.time()
#             label_g, conf_g, probs_g, _ = _gate_jump_and_decide(probs.copy(), arr_raw)
#             probs_p = (
#                 _apply_yaw_prior(device_id, probs_g, arr_raw)
#                 if state.get("yaw_assist", True)
#                 else probs_g
#             )

#             probs_s = _ema_update(device_id, probs_p)

#             ps = float(np.sum(probs_s))
#             if ps > 1e-9:
#                 probs_s = probs_s / ps

#             idx = int(np.argmax(probs_s))
#             classes = list(state["le"].classes_)
#             label = classes[idx]
#             conf = float(probs_s[idx])
#             cmd = _debounced_command(device_id, label, conf, now_ts, arr_raw)

#             latest_sample = state["recent"][device_id][-1] if state["recent"][device_id] else {}
#             ctrl = update_control(state, device_id, arr_raw, latest_sample, now_ts, cmd)

#             mean = arr_raw.mean(axis=0).tolist()
#             pred = {
#                 "label": label,
#                 "command": cmd or "NONE",
#                 "confidence": conf,
#                 "probs": {classes[i]: float(probs_s[i]) for i in range(len(classes))},
#                 "raw_latest": latest_sample,
#                 "raw_mean": {
#                     "gx": mean[0], "gy": mean[1], "gz": mean[2],
#                     "ax": mean[3], "ay": mean[4], "az": mean[5],
#                 },
#                 "timestamp": now_ts,
#                 "window": state["eff_window"],
#                 "control": ctrl,
#             }

#         if pred is not None:
#             state["latest"][device_id] = pred
            
#             # Publish result to MQTT
#             response = {
#                 "ok": True,
#                 "device_id": device_id,
#                 "status": "ready",
#                 "command": pred.get("command", "NONE"),
#                 "label": pred.get("label", "NONE"),
#                 "confidence": pred.get("confidence", 0.0),
#                 "control": pred.get("control", {}),
#                 "timestamp": pred.get("timestamp", time.time())
#             }
#             client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
            
#             # Check if we should send sleep command
#             if pred.get("command") == "JUMP":
#                 sleep_msg = {"device_id": device_id, "sleep": True}
#                 client.publish(TOPIC_SLEEP, json.dumps(sleep_msg), qos=1)
#                 print(f"[INFO] Sent sleep command for {device_id}")


# def on_connect(client, userdata, flags, rc):
#     """Callback when connected to MQTT broker"""
#     print(f"[MQTT] Connected with result code {rc}")
#     client.subscribe(TOPIC_IMU_IN)
#     print(f"[MQTT] Subscribed to {TOPIC_IMU_IN}")


# def on_message(client, userdata, message):
#     """Handle incoming MQTT messages"""
#     try:
#         topic = message.topic
#         payload_str = message.payload.decode('utf-8')
        
#         if topic == TOPIC_IMU_IN:
#             payload = json.loads(payload_str)
#             process_imu_data(payload, client)
            
#     except json.JSONDecodeError as e:
#         print(f"[ERROR] JSON decode error: {e}")
#     except Exception as e:
#         print(f"[ERROR] Error processing message: {e}")
#         import traceback
#         traceback.print_exc()


# @atexit.register
# def _on_exit_flush_logs():
#     try:
#         if state.get("log_rows"):
#             log_dir = Path(state.get("log_dir") or ".")
#             log_dir.mkdir(parents=True, exist_ok=True)
#             ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#             path = log_dir / f"predictions_log_{ts}.csv"
            
#             base_cols = ["timestamp", "device_id", "label", "command", "confidence"]
#             prob_cols = [f"prob_{c}" for c in getattr(state.get("le", None), "classes_", [])]
#             raw_cols = [f"raw_{k}" for k in ["gx", "gy", "gz", "ax", "ay", "az"]]
#             cols = base_cols + prob_cols + raw_cols
            
#             with open(path, "w", newline="") as f:
#                 w = csv.DictWriter(f, fieldnames=cols)
#                 w.writeheader()
#                 for r in state["log_rows"]:
#                     w.writerow({k: r.get(k) for k in cols})
            
#             print(f"[INFO] Wrote prediction log to {path}")
#     except Exception as e:
#         print(f"[WARN] Failed to write prediction log on exit: {e}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--artifacts", required=True)
#     ap.add_argument("--threshold", type=float, default=0.6)
#     ap.add_argument("--broker_ip", default=BROKER_IP)
#     ap.add_argument("--broker_port", type=int, default=BROKER_PORT)
#     ap.add_argument("--window", type=int, default=0)
#     ap.add_argument("--debounce", type=int, default=2)
#     ap.add_argument("--jump_gate", type=float, default=12.0)
#     ap.add_argument("--auto_calib", type=int, default=0)
#     ap.add_argument("--jump_reset_delay_ms", type=int, default=1500)
    
#     args = ap.parse_args()
    
#     # Initialize state
#     state["debounce_k"] = max(1, int(args.debounce))
#     state["min_acc_for_jump"] = float(args.jump_gate)
#     state["auto_calib"] = int(args.auto_calib)
#     state["jump_reset_delay_ms"] = int(args.jump_reset_delay_ms)
    
#     # Initialize model
#     init_model(Path(args.artifacts), threshold=args.threshold, window_override=args.window)
    
#     # Setup MQTT client
#     mqtt_client = mqtt.Client(client_id="imu_lstm_server")
#     mqtt_client.on_connect = on_connect
#     mqtt_client.on_message = on_message
    
#     print(f"[INFO] Connecting to MQTT broker at {args.broker_ip}:{args.broker_port}")
#     mqtt_client.connect(args.broker_ip, args.broker_port, 60)
    
#     print(f"[INFO] LSTM Model ready | window={state['eff_window']}")
#     print(f"[INFO] Publishing to: {TOPIC_IMU_OUT}, {TOPIC_SLEEP}")
#     print("[INFO] Ready to process IMU data\n")
    
#     try:
#         mqtt_client.loop_forever()
#     except KeyboardInterrupt:
#         print("\n[INFO] Shutting down...")
#         mqtt_client.disconnect()
#         print("[INFO] Goodbye!")

import paho.mqtt.client as mqtt
import time
import argparse, sys, json, os, atexit, csv
import numpy as np, torch
from pathlib import Path
from typing import Dict, Any
from collections import deque
from datetime import datetime
from threading import Lock

# Make "src" importable
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import load_joblib, load_json
from models.lstm_v2 import LSTMIMU
from realtime.sliding_window import SlidingWindow
from realtime.motion_control import ensure_control_state, update_control

# MQTT Configuration
BROKER_IP = "10.180.115.152"  # Update this to your broker IP
BROKER_PORT = 1883
TOPIC_IMU_IN = "/imu"
TOPIC_IMU_OUT = "/imu_out"
TOPIC_SLEEP = "/eve_sleep"

YAW_IDX = {"gx": 0, "gy": 1, "gz": 2}

state = {
    "scaler": None,
    "le": None,
    "cfg": None,
    "model": None,
    "device": None,
    "eff_window": None,
    "artifacts": None,
    "threshold": 0.6,
    "per_class_threshold": {},
    "debounce_k": 2,
    "min_acc_for_jump": 12.0,
    "debounce_map": {"default": 3, "straight": 2},
    "threshold_map": {"default": 0.6, "straight": 0.55, "left": 0.7, "right": 0.7},
    "quiet_gyro": 0.06,
    "quiet_ms": 250,
    "last_quiet": {},
    "turn_fast_thr": 0.90,
    "turn_min_gyro": 0.10,
    "ema_probs": {},
    "ema_alpha": 0.6,
    "yaw_assist": True,
    "yaw_axis": "gx",
    "yaw_sign": 1,
    "yaw_min": 0.12,
    "yaw_window": 12,
    "yaw_prior_k": 2.0,
    "yaw_prior_min": 0.08,
    "yaw_prior_max": 0.35,
    "probs_s": {},
    "jump_hold_ms": 140,
    "jump_lock_until": {},
    "ctl": {},
    "max_speed": 1.0,
    "speed_alpha": 0.25,
    "yaw_alpha": 0.30,
    "max_yaw_rate_dps": 180.0,
    "turn_mix": 0.6,
    "acc_to_speed_gain": 0.08,
    "acc_bias_alpha": 0.01,
    "speed_floor": 0.0,
    "auto_calib": 0,
    "auto_calib_done": {},
    "turn_guard_ms": 150.0,
    "last_turn": {},
    "windows": {},
    "latest": {},
    "recent": {},
    "rot": {},
    "g_ref": None,
    "hist": {},
    "current_cmd": {},
    "classes": [],
    "log_rows": [],
    "log_dir": None,
    "auto_calib_on_first_data": True,
    "calib_done_devices": set(),
    "jump_reset_delay_ms": 1500,
    "jump_detected_at": {},
}

state_lock = Lock()

AXIS_MAP = {"idx": [0, 1, 2, 3, 4, 5], "sign": [1, 1, 1, 1, 1, 1]}


def load_axis_map(artifacts: Path):
    path = artifacts / "axis_map.json"
    if path.exists():
        try:
            cfg = json.load(open(path))
            idx = cfg.get("idx", AXIS_MAP["idx"])
            sign = cfg.get("sign", AXIS_MAP["sign"])
            if len(idx) == 6 and len(sign) == 6:
                return {"idx": [int(i) for i in idx], "sign": [int(s) for s in sign]}
        except Exception:
            pass
    return AXIS_MAP


def apply_map(parts):
    out = [0.0] * 6
    for i in range(6):
        out[i] = AXIS_MAP["sign"][i] * parts[AXIS_MAP["idx"][i]]
    return out


def load_gravity_ref(artifacts: Path):
    path = artifacts / "gravity_ref.json"
    if path.exists():
        try:
            j = json.load(open(path))
            g = np.array(j.get("g_ref", [0, 9.81, 0]), dtype=float)
            if g.shape == (3,):
                return g
        except Exception:
            pass
    return np.array([0.0, 9.81, 0.0], dtype=float)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def rot_from_vectors(a, b):
    a_u, b_u = normalize(a), normalize(b)
    v = np.cross(a_u, b_u)
    c = np.dot(a_u, b_u)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


def apply_rotation(parts, device_id: str):
    R = state["rot"].get(device_id, None)
    if R is None:
        return parts
    g = np.array(parts[0:3], dtype=float)
    a = np.array(parts[3:6], dtype=float)
    g_r = (R @ g).tolist()
    a_r = (R @ a).tolist()
    return g_r + a_r


def _rot_path(device_id: str) -> Path:
    d = Path(state["artifacts"]) / "calib"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"rot_{device_id}.json"


def save_rot(device_id: str, R):
    p = _rot_path(device_id)
    with open(p, "w") as f:
        json.dump({"R": (R.tolist() if hasattr(R, "tolist") else R)}, f)


def load_rot(device_id: str):
    p = _rot_path(device_id)
    if p.exists():
        try:
            j = json.load(open(p))
            return np.array(j["R"], dtype=float)
        except Exception:
            pass
    return None


def init_model(artifacts: Path, threshold: float, window_override: int = 0):
    global AXIS_MAP
    AXIS_MAP = load_axis_map(artifacts)
    scaler = load_joblib(artifacts / "scaler.joblib")
    le = load_joblib(artifacts / "label_encoder.joblib")
    cfg = load_json(artifacts / "config.json")["config"]
    eff_window = (
        window_override
        if window_override > 0
        else cfg.get("effective_window", cfg.get("window", 150))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMIMU(
        input_size=6,
        hidden_size=cfg["hidden"],
        num_layers=cfg["layers"],
        dropout=cfg["dropout"],
        num_classes=len(le.classes_),
        bidirectional=cfg["bidirectional"],
    ).to(device)
    model.load_state_dict(
        torch.load(artifacts / "lstm_v2_best.pth", map_location=device)
    )
    model.eval()
    state.update(
        dict(
            scaler=scaler,
            le=le,
            cfg=cfg,
            model=model,
            device=device,
            eff_window=eff_window,
            threshold=threshold,
            artifacts=str(artifacts),
        )
    )
    state["g_ref"] = load_gravity_ref(artifacts)
    log_dir = Path(artifacts) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state["log_dir"] = str(log_dir)
    print("[INFO] Axis map idx=", AXIS_MAP["idx"], "sign=", AXIS_MAP["sign"])
    print("[INFO] g_ref =", state["g_ref"].tolist())


def ensure_window(device_id: str):
    if device_id not in state["windows"]:
        state["windows"][device_id] = SlidingWindow(state["eff_window"], n_feat=6)
    if device_id not in state["recent"]:
        state["recent"][device_id] = deque(maxlen=300)
    if device_id not in state["rot"]:
        R = load_rot(device_id)
        state["rot"][device_id] = R if R is not None else np.eye(3)
    if device_id not in state["hist"]:
        state["hist"][device_id] = deque(maxlen=8)
    if device_id not in state["last_quiet"]:
        state["last_quiet"][device_id] = 0.0
    ensure_control_state(state, device_id)


def _gate_jump_and_decide(probs: np.ndarray, arr_raw: np.ndarray):
    classes = list(state["le"].classes_)
    idx = int(np.argmax(probs))
    if classes[idx].lower() == "jump":
        acc = arr_raw[:, 3:6]
        g_hat = normalize(state["g_ref"])
        a_proj_abs = np.abs(acc @ g_hat)
        a_p90 = float(np.percentile(a_proj_abs, 90.0))
        if a_p90 < float(state["min_acc_for_jump"]):
            probs[idx] = 0.0
            idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs, idx


def _apply_yaw_prior(device_id: str, probs: np.ndarray, arr_raw: np.ndarray) -> np.ndarray:
    classes = list(state["le"].classes_)
    try:
        li = classes.index("left")
        ri = classes.index("right")
    except ValueError:
        return probs
    yaw_idx = YAW_IDX.get(state.get("yaw_axis", "gx"), 0)
    n = max(3, min(int(state.get("yaw_window", 12)), arr_raw.shape[0]))
    yaw = arr_raw[-n:, yaw_idx].mean() * float(state.get("yaw_sign", 1))
    a = float(state.get("yaw_prior_min", 0.08))
    b = float(state.get("yaw_prior_max", 0.35))
    mag = max(0.0, min(1.0, (abs(yaw) - a) / max(1e-6, (b - a))))
    if mag <= 0.0:
        return probs
    k = float(state.get("yaw_prior_k", 2.0))
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    if yaw > 0:
        logits[ri] += k * mag
    elif yaw < 0:
        logits[li] += k * mag
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _ema_update(device_id: str, probs: np.ndarray) -> np.ndarray:
    alpha = float(state.get("ema_alpha", 0.6))
    prev = state["ema_probs"].get(device_id, None)
    out = probs if prev is None else (alpha * probs + (1.0 - alpha) * prev)
    s = float(np.sum(out))
    if s > 1e-9:
        out = out / s
    state["ema_probs"][device_id] = out
    state["probs_s"][device_id] = out
    return out


def _debounced_command(device_id: str, label: str, confidence: float, now_ts: float, arr_raw: np.ndarray) -> str:
    g = arr_raw[:, :3]
    g_mag = float(np.sqrt((g**2).sum(axis=1)).mean())
    if g_mag < float(state["quiet_gyro"]):
        state["last_quiet"][device_id] = now_ts

    th_map = state["threshold_map"]
    th_def = float(th_map.get("default", state["threshold"]))
    th_straight = float(th_map.get("straight", th_def))
    th_label = float(th_map.get(label.lower(), th_def))

    jump_detected_at = state.get("jump_detected_at", {}).get(device_id, 0.0)
    reset_delay_ms = float(state.get("jump_reset_delay_ms", 1500))
    
    if jump_detected_at > 0:
        elapsed_ms = (now_ts - jump_detected_at) * 1000.0
        if elapsed_ms >= reset_delay_ms:
            print(f"[INFO] Jump reset period over for {device_id}, clearing state")
            _reset_device_state(device_id, keep_calibration=True)
            return "NONE"
        else:
            return "JUMP"

    if label.lower() == "straight" and confidence >= th_straight:
        state["current_cmd"][device_id] = "STRAIGHT"
        state["hist"][device_id].clear()
        state.get("last_turn", {}).pop(device_id, None)
        return "STRAIGHT"

    classes = list(state["le"].classes_)
    try:
        j_idx = classes.index("jump")
    except ValueError:
        j_idx = None

    lock_until = state["jump_lock_until"].get(device_id, 0.0)
    if now_ts < float(lock_until):
        state["current_cmd"][device_id] = "JUMP"
        return "JUMP"

    if j_idx is not None:
        probs_s = state["probs_s"].get(device_id)
        p_jump = float(probs_s[j_idx]) if probs_s is not None else 0.0
        th_jump = float(th_map.get("jump", th_def))
        acc = arr_raw[:, 3:6]
        g_hat = normalize(state["g_ref"])
        a_p90 = float(np.percentile(np.abs(acc @ g_hat), 90.0))
        if (p_jump >= float(state["turn_fast_thr"])) or (
            p_jump >= th_jump and a_p90 >= float(state["min_acc_for_jump"])
        ):
            state["current_cmd"][device_id] = "JUMP"
            state["jump_lock_until"][device_id] = (
                now_ts + float(state["jump_hold_ms"]) / 1000.0
            )
            state["jump_detected_at"][device_id] = now_ts
            print(f"[INFO] JUMP detected for {device_id} at {now_ts}")
            return "JUMP"

    if label.lower() != "straight":
        if confidence >= float(state["turn_fast_thr"]) and g_mag >= float(state["turn_min_gyro"]):
            guard_ms = float(state.get("turn_guard_ms", 150.0))
            last = state.get("last_turn", {}).get(device_id, {"dir": None, "ts": 0.0})
            opp = "LEFT" if label.upper() == "RIGHT" else "RIGHT"
            if (
                last
                and last.get("dir") == opp
                and (now_ts - float(last.get("ts", 0.0))) * 1000.0 < guard_ms
            ):
                pass
            else:
                state["current_cmd"][device_id] = label.upper()
                state["hist"][device_id].clear()
                state.setdefault("last_turn", {})[device_id] = {
                    "dir": label.upper(),
                    "ts": now_ts,
                }
                return label.upper()

    k_map = state["debounce_map"]
    k_def = int(k_map.get("default", state["debounce_k"]))
    k = int(k_map.get(label.lower(), k_def))
    hist = state["hist"][device_id]
    hist.append(label)

    if confidence < th_label:
        return state["current_cmd"].get(device_id, "NONE")

    last_k = list(hist)[-k:]
    if len(last_k) == k and all(l == label for l in last_k):
        state["current_cmd"][device_id] = label.upper()
        if label.lower() != "straight":
            state.setdefault("last_turn", {})[device_id] = {
                "dir": label.upper(),
                "ts": now_ts,
            }
        else:
            state.get("last_turn", {}).pop(device_id, None)
    return state["current_cmd"].get(device_id, "NONE")


def _reset_device_state(device_id: str, keep_calibration: bool = True):
    saved_rot = None
    saved_calib_status = False
    
    if keep_calibration:
        saved_rot = state["rot"].get(device_id)
        saved_calib_status = device_id in state.get("calib_done_devices", set())
    
    state["windows"].pop(device_id, None)
    state["recent"].pop(device_id, None)
    state["hist"].pop(device_id, None)
    state["last_quiet"].pop(device_id, None)
    state["latest"].pop(device_id, None)
    state["current_cmd"].pop(device_id, None)
    state["jump_lock_until"].pop(device_id, None)
    state["jump_detected_at"].pop(device_id, None)
    state["ema_probs"].pop(device_id, None)
    state["probs_s"].pop(device_id, None)
    state.get("last_turn", {}).pop(device_id, None)
    state.get("ctl", {}).pop(device_id, None)
    state["auto_calib_done"].pop(device_id, None)
    
    if keep_calibration and saved_rot is not None:
        state["rot"][device_id] = saved_rot
        if saved_calib_status:
            state.setdefault("calib_done_devices", set()).add(device_id)
    
    ensure_window(device_id)
    print(f"[INFO] Reset state for {device_id} (calibration preserved: {keep_calibration})")


def calibrate_gravity(device_id: str, client):
    ensure_window(device_id)
    win = state["windows"][device_id]
    if not win.ready():
        return {"ok": False, "error": "window not ready"}
    arr = win.array()
    g_live = arr[:, 3:6].mean(axis=0)
    g_ref = state["g_ref"]
    R = rot_from_vectors(g_live, g_ref)
    state["rot"][device_id] = R
    save_rot(device_id, R)
    return {
        "ok": True,
        "g_live": g_live.tolist(),
        "g_ref": g_ref.tolist(),
        "R": R.tolist(),
    }


def auto_yaw(device_id: str):
    win = state["windows"].get(device_id)
    if not win or not win.ready():
        return {"ok": False, "error": "window not ready"}
    arr = win.array()
    stds = arr[:, 0:3].std(axis=0)
    axes = ["gx", "gy", "gz"]
    axis = axes[int(np.argmax(stds))]
    state["yaw_axis"] = axis
    return {
        "ok": True,
        "axis": axis,
        "sign": state["yaw_sign"],
        "std": {axes[i]: float(stds[i]) for i in range(3)},
    }


def _append_log_row(
    device_id: str,
    ts: float,
    label: str,
    command: str,
    confidence: float,
    probs: Dict[str, float],
    raw_latest: Dict[str, float],
):
    row = {
        "timestamp": ts,
        "device_id": device_id,
        "label": label,
        "command": command,
        "confidence": confidence,
    }
    row["prob_sum"] = float(sum(probs.values()))
    for c in state["le"].classes_:
        row[f"prob_{c}"] = float(probs.get(c, 0.0))
    if raw_latest:
        for k in ["gx", "gy", "gz", "ax", "ay", "az"]:
            row[f"raw_{k}"] = float(raw_latest.get(k, 0.0))
    state["log_rows"].append(row)


def process_imu_data(payload: Dict[str, Any], client):
    """Process incoming IMU data and publish results"""
    with state_lock:
        device_id = str(payload.get("device_id", "imu01"))
        samples = payload.get("samples", [])
        
        ensure_window(device_id)
        win = state["windows"][device_id]
        
        for s in samples:
            parts = [
                float(s["gx"]), float(s["gy"]), float(s["gz"]),
                float(s["ax"]), float(s["ay"]), float(s["az"]),
            ]
            parts = apply_map(parts)
            parts = apply_rotation(parts, device_id)
            ts = int(s.get("ts", time.time() * 1000))
            latest_sample = {
                "ts": ts,
                "gx": parts[0], "gy": parts[1], "gz": parts[2],
                "ax": parts[3], "ay": parts[4], "az": parts[5],
            }
            state["recent"][device_id].append(latest_sample)
            win.push(parts)
        
        # Check calibration
        if state.get("auto_calib", 0) and device_id not in state.get("calib_done_devices", set()):
            if win.ready():
                print(f"[INFO] Performing auto-calibration for {device_id}")
                calibrate_gravity(device_id, client)
                time.sleep(0.1)
                auto_yaw(device_id)
                state.setdefault("calib_done_devices", set()).add(device_id)
                
                response = {
                    "ok": True,
                    "status": "calibrating",
                    "message": "Calibration complete"
                }
                client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
                return
            else:
                response = {
                    "ok": True,
                    "status": "calibrating",
                    "message": "Collecting calibration data...",
                    "progress": f"{win.count}/{state['eff_window']}"
                }
                client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
                return
        
        # Normal processing
        pred = None
        if win.ready():
            arr_raw = win.array()
            arr = arr_raw[None, ...]
            arr_s = (
                state["scaler"]
                .transform(arr.reshape(-1, arr.shape[-1]))
                .reshape(arr.shape)
            )
            xb = torch.tensor(arr_s, dtype=torch.float32).to(state["device"])
            with torch.no_grad():
                logits = state["model"](xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            now_ts = time.time()
            label_g, conf_g, probs_g, _ = _gate_jump_and_decide(probs.copy(), arr_raw)
            probs_p = (
                _apply_yaw_prior(device_id, probs_g, arr_raw)
                if state.get("yaw_assist", True)
                else probs_g
            )

            probs_s = _ema_update(device_id, probs_p)

            ps = float(np.sum(probs_s))
            if ps > 1e-9:
                probs_s = probs_s / ps

            idx = int(np.argmax(probs_s))
            classes = list(state["le"].classes_)
            label = classes[idx]
            conf = float(probs_s[idx])
            cmd = _debounced_command(device_id, label, conf, now_ts, arr_raw)

            latest_sample = state["recent"][device_id][-1] if state["recent"][device_id] else {}
            ctrl = update_control(state, device_id, arr_raw, latest_sample, now_ts, cmd)

            mean = arr_raw.mean(axis=0).tolist()
            pred = {
                "label": label,
                "command": cmd or "NONE",
                "confidence": conf,
                "probs": {classes[i]: float(probs_s[i]) for i in range(len(classes))},
                "raw_latest": latest_sample,
                "raw_mean": {
                    "gx": mean[0], "gy": mean[1], "gz": mean[2],
                    "ax": mean[3], "ay": mean[4], "az": mean[5],
                },
                "timestamp": now_ts,
                "window": state["eff_window"],
                "control": ctrl,
            }
            
            _append_log_row(
                device_id,
                pred["timestamp"],
                label,
                pred["command"],
                pred["confidence"],
                pred["probs"],
                latest_sample,
            )

        if pred is not None:
            state["latest"][device_id] = pred
            
            # Publish result to MQTT
            response = {
                "ok": True,
                "device_id": device_id,
                "status": "ready",
                "command": pred.get("command", "NONE"),
                "label": pred.get("label", "NONE"),
                "confidence": pred.get("confidence", 0.0),
                "control": pred.get("control", {}),
                "timestamp": pred.get("timestamp", time.time())
            }
            client.publish(TOPIC_IMU_OUT, json.dumps(response), qos=1)
            
            # Check if we should send sleep command
            if pred.get("command") == "JUMP":
                sleep_msg = {"device_id": device_id, "sleep": True}
                client.publish(TOPIC_SLEEP, json.dumps(sleep_msg), qos=1)
                print(f"[INFO] Sent sleep command for {device_id}")


def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(TOPIC_IMU_IN)
    print(f"[MQTT] Subscribed to {TOPIC_IMU_IN}")


def on_message(client, userdata, message):
    """Handle incoming MQTT messages"""
    try:
        topic = message.topic
        payload_str = message.payload.decode('utf-8')
        
        if topic == TOPIC_IMU_IN:
            payload = json.loads(payload_str)
            process_imu_data(payload, client)
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error: {e}")
    except Exception as e:
        print(f"[ERROR] Error processing message: {e}")
        import traceback
        traceback.print_exc()


@atexit.register
def _on_exit_flush_logs():
    try:
        if state.get("log_rows"):
            log_dir = Path(state.get("log_dir") or ".")
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"predictions_log_{ts}.csv"
            
            base_cols = ["timestamp", "device_id", "label", "command", "confidence"]
            prob_cols = [f"prob_{c}" for c in getattr(state.get("le", None), "classes_", [])]
            raw_cols = [f"raw_{k}" for k in ["gx", "gy", "gz", "ax", "ay", "az"]]
            cols = base_cols + prob_cols + raw_cols
            
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in state["log_rows"]:
                    w.writerow({k: r.get(k) for k in cols})
            
            print(f"[INFO] Wrote prediction log to {path}")
    except Exception as e:
        print(f"[WARN] Failed to write prediction log on exit: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--broker_ip", default=BROKER_IP)
    ap.add_argument("--broker_port", type=int, default=BROKER_PORT)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--debounce", type=int, default=2)
    ap.add_argument("--jump_gate", type=float, default=12.0)
    ap.add_argument("--threshold_jump", type=float, default=-1.0)
    ap.add_argument("--debounce_jump", type=int, default=1)
    ap.add_argument("--jump_hold_ms", type=int, default=140)
    ap.add_argument("--debounce_straight", type=int, default=2)
    ap.add_argument("--threshold_straight", type=float, default=-1.0)
    ap.add_argument("--threshold_left", type=float, default=-1.0)
    ap.add_argument("--threshold_right", type=float, default=-1.0)
    ap.add_argument("--quiet_gyro", type=float, default=0.06)
    ap.add_argument("--quiet_ms", type=int, default=250)
    ap.add_argument("--turn_fast_thr", type=float, default=0.90)
    ap.add_argument("--turn_min_gyro", type=float, default=0.10)
    ap.add_argument("--ema_alpha", type=float, default=0.6)
    ap.add_argument("--yaw_prior_k", type=float, default=2.0)
    ap.add_argument("--yaw_prior_min", type=float, default=0.08)
    ap.add_argument("--yaw_prior_max", type=float, default=0.35)
    ap.add_argument("--max_speed", type=float, default=1.0)
    ap.add_argument("--speed_alpha", type=float, default=0.25)
    ap.add_argument("--yaw_alpha", type=float, default=0.30)
    ap.add_argument("--max_yaw_rate_dps", type=float, default=180.0)
    ap.add_argument("--turn_mix", type=float, default=0.6)
    ap.add_argument("--acc_to_speed_gain", type=float, default=0.08)
    ap.add_argument("--acc_bias_alpha", type=float, default=0.01)
    ap.add_argument("--speed_floor", type=float, default=0.0)
    ap.add_argument("--auto_calib", type=int, default=0)
    ap.add_argument("--turn_guard_ms", type=float, default=150.0)
    ap.add_argument("--turn_release_thr", type=float, default=0.0)
    ap.add_argument("--straight_margin", type=float, default=0.0)
    ap.add_argument("--jump_reset_delay_ms", type=int, default=1500)
    
    args = ap.parse_args()
    
    # Initialize state with all parameters
    state["debounce_k"] = max(1, int(args.debounce))
    state["min_acc_for_jump"] = float(args.jump_gate)
    state["debounce_map"]["default"] = max(1, int(args.debounce))
    state["debounce_map"]["straight"] = max(1, int(args.debounce_straight))
    state["debounce_map"]["jump"] = max(1, int(args.debounce_jump))
    state["threshold_map"]["default"] = float(args.threshold)
    state["threshold_map"]["straight"] = (
        float(args.threshold_straight)
        if args.threshold_straight > 0
        else max(0.0, float(args.threshold) - 0.05)
    )
    state["threshold_map"]["left"] = (
        args.threshold_left if args.threshold_left > 0 else float(args.threshold)
    )
    state["threshold_map"]["right"] = (
        args.threshold_right if args.threshold_right > 0 else float(args.threshold)
    )
    state["threshold_map"]["jump"] = (
        args.threshold_jump if args.threshold_jump > 0 else float(args.threshold)
    )
    state["quiet_gyro"] = float(args.quiet_gyro)
    state["quiet_ms"] = int(args.quiet_ms)
    state["turn_fast_thr"] = float(args.turn_fast_thr)
    state["turn_min_gyro"] = float(args.turn_min_gyro)
    state["ema_alpha"] = float(args.ema_alpha)
    state["yaw_prior_k"] = float(args.yaw_prior_k)
    state["yaw_prior_min"] = float(args.yaw_prior_min)
    state["yaw_prior_max"] = float(args.yaw_prior_max)
    state["jump_hold_ms"] = int(args.jump_hold_ms)
    state["max_speed"] = float(args.max_speed)
    state["speed_alpha"] = float(args.speed_alpha)
    state["yaw_alpha"] = float(args.yaw_alpha)
    state["max_yaw_rate_dps"] = float(args.max_yaw_rate_dps)
    state["turn_mix"] = float(args.turn_mix)
    state["acc_to_speed_gain"] = float(args.acc_to_speed_gain)
    state["acc_bias_alpha"] = float(args.acc_bias_alpha)
    state["speed_floor"] = float(args.speed_floor)
    state["auto_calib"] = int(args.auto_calib)
    state["turn_guard_ms"] = float(args.turn_guard_ms)
    if args.turn_release_thr > 0:
        state["turn_release_thr"] = float(args.turn_release_thr)
    if args.straight_margin > 0:
        state["straight_margin"] = float(args.straight_margin)
    state["jump_reset_delay_ms"] = int(args.jump_reset_delay_ms)
    
    # Initialize model
    init_model(Path(args.artifacts), threshold=args.threshold, window_override=args.window)
    
    # Setup MQTT client
    mqtt_client = mqtt.Client(client_id="imu_lstm_server")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    print(f"[INFO] Connecting to MQTT broker at {args.broker_ip}:{args.broker_port}")
    mqtt_client.connect(args.broker_ip, args.broker_port, 60)
    
    print(f"[INFO] LSTM Model ready | window={state['eff_window']}")
    print(f"[INFO] Publishing to: {TOPIC_IMU_OUT}, {TOPIC_SLEEP}")
    print("[INFO] Ready to process IMU data\n")
    
    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        mqtt_client.disconnect()
        print("[INFO] Goodbye!")