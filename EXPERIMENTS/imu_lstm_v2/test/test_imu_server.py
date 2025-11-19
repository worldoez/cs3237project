from flask import Flask, request, jsonify
import time

app = Flask(__name__)


ctl = {}


@app.get("/control")
def get_distance():
    global ctl

    latest = ctl

    return {
        "ok": True,
        "device_id": "imu01",
        "command": latest.get("command", "NONE"),
        "label": latest.get("label", "NONE"),
        "confidence": float(latest.get("confidence", 0.0)),
        "speed": float(ctl.get("speed_lp", 0.0)),
        "yaw_rate_dps": float(ctl.get("yaw_rate_lp_dps", 0.0)),
        "turn_angle_deg": float(ctl.get("angle_deg", 0.0)),
        "left_speed": float(ctl.get("left_speed", 0.0)),
        "right_speed": float(ctl.get("right_speed", 0.0)),
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
