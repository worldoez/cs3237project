import math
import numpy as np

# local yaw index map (match http_ingest_server)
YAW_IDX = {"gx": 0, "gy": 1, "gz": 2}
RAD2DEG = 180.0 / math.pi

def _proj_plane(a: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    # Remove component along n_hat
    return a - np.dot(a, n_hat) * n_hat

def ensure_control_state(state: dict, device_id: str):
    ctl = state.setdefault("ctl", {})
    if device_id not in ctl:
        ctl[device_id] = {
            "last_ts_ms": 0,
            "yaw_rate_lp_dps": 0.0,
            "speed_lp": 0.0,
            "angle_deg": 0.0,
            "acc_plane_bias": 0.0,
            "left_speed": 0.0,
            "right_speed": 0.0,
        }

def update_control(state: dict, device_id: str, arr_raw: np.ndarray, latest_sample: dict, now_ts: float, command: str) -> dict:
    """
    Compute smoothed speed (0..1), yaw_rate (deg/s), turn angle since turn onset (deg),
    and differential motor mixing (left/right in -1..1).
    """
    ensure_control_state(state, device_id)
    ctl = state["ctl"][device_id]

    # Params
    yaw_axis = state.get("yaw_axis", "gx")
    yaw_sign = float(state.get("yaw_sign", 1))
    yaw_alpha = float(state.get("yaw_alpha", 0.30))
    speed_alpha = float(state.get("speed_alpha", 0.25))
    acc_gain = float(state.get("acc_to_speed_gain", 0.08))
    acc_bias_alpha = float(state.get("acc_bias_alpha", 0.01))
    max_speed = float(state.get("max_speed", 1.0))
    max_yaw_dps = float(state.get("max_yaw_rate_dps", 180.0))
    turn_mix = float(state.get("turn_mix", 0.6))
    speed_floor = float(state.get("speed_floor", 0.0))
    quiet_ms = float(state.get("quiet_ms", 250))

    # Latest vectors
    a = np.array(arr_raw[-1, 3:6], dtype=float)
    g_hat = _safe_normalize(np.array(state.get("g_ref"), dtype=float))
    a_plane = _proj_plane(a, g_hat)
    a_plane_mag = float(np.linalg.norm(a_plane))

    # Learn plane accel bias while quiet to reduce drift
    last_quiet = float(state.get("last_quiet", {}).get(device_id, 0.0))
    if (now_ts - last_quiet) * 1000.0 <= quiet_ms:
        ctl["acc_plane_bias"] = (1.0 - acc_bias_alpha) * ctl["acc_plane_bias"] + acc_bias_alpha * a_plane_mag

    cmd_upper = (command or "").upper()

    # Speed estimate from plane accel magnitude (smooth)
    speed_raw = max(0.0, a_plane_mag - ctl["acc_plane_bias"]) * acc_gain
    speed_raw = max(speed_floor, min(max_speed, speed_raw))

    # Optional: zero speed during jump for safety
    if cmd_upper == "JUMP":
        speed_raw = 0.0

    speed_lp = speed_alpha * speed_raw + (1.0 - speed_alpha) * ctl["speed_lp"]

    # Yaw rate (deg/s) from selected gyro axis
    yaw_idx = YAW_IDX.get(yaw_axis, 0)
    yaw_rate_rad = float(arr_raw[-1, yaw_idx]) * yaw_sign
    yaw_rate_dps = yaw_rate_rad * RAD2DEG
    yaw_rate_lp_dps = yaw_alpha * yaw_rate_dps + (1.0 - yaw_alpha) * ctl["yaw_rate_lp_dps"]

    # Integrate turn angle only while turning (LEFT/RIGHT)
    ts_ms = int(latest_sample.get("ts", round(now_ts * 1000)))
    last_ts_ms = int(ctl.get("last_ts_ms", ts_ms))
    dt_s = max(0.0, (ts_ms - last_ts_ms) / 1000.0)

    # Reset angle when entering STRAIGHT or JUMP
    cmd_upper = (command or "").upper()
    if cmd_upper not in ("LEFT", "RIGHT"):
        # decay angle toward 0 slowly to avoid stale value
        ctl["angle_deg"] = 0.0
    else:
        # signed integration (yaw_sign already applied; sign of LEFT/RIGHT is in yaw rate)
        ctl["angle_deg"] += yaw_rate_lp_dps * dt_s

    # Differential mixing for a simple driver (normalized -1..1)
    yaw_norm = 0.0 if max_yaw_dps <= 1e-6 else max(-1.0, min(1.0, yaw_rate_lp_dps / max_yaw_dps))
    left = max(-1.0, min(1.0, speed_lp - turn_mix * yaw_norm))
    right = max(-1.0, min(1.0, speed_lp + turn_mix * yaw_norm))

    # Persist
    ctl.update({
        "last_ts_ms": ts_ms,
        "yaw_rate_lp_dps": yaw_rate_lp_dps,
        "speed_lp": speed_lp,
        "left_speed": left,
        "right_speed": right,
    })

    return {
        "speed": float(speed_lp),                   # 0..1 normalized
        "yaw_rate_dps": float(yaw_rate_lp_dps),     # deg/s
        "turn_angle_deg": float(ctl["angle_deg"]),  # signed, since last turn onset
        "left_speed": float(left),                  # -1..1
        "right_speed": float(right),                # -1..1
        "timestamp": now_ts,
    }

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-9)