from flask import Flask, request, jsonify
import time

app = Flask(__name__)

distance_data = {}  # in memory temp storage


@app.post("/distance")
def post_distance():
    global distance_data

    data = request.get_json() or {}

    if "distance" not in data:
        return {"ok": False, "error": "Missing [distance] field in request."}, 400

    distance_data = {
        "device_id": data.get("device_id", "unknown_device"),
        "distance": float(data.get("distance", -1.0)),
        "is_apriltag_present": bool(data.get("is_apriltag_present", False)),
        "apriltag_center": data.get("apriltag_center", [-1, -1]),
        "timestamp": data.get("timestamp", time.time()),
    }
    print("Received data:", data)

    return {
        "ok": True,
        "message": "Distance val successfully stored.",
        "data": distance_data,
    }


# returns stored distance data, just retreives last recorded data
@app.get("/getDistance")
def get_distance():
    global distance_data

    if not distance_data:  # empty cuz post endpoint not hit yet
        return {"ok": False, "error": "404, Distance probably not recorded"}, 404

    latest = distance_data

    return {
        "ok": True,
        "data": {
            "device_id": latest.get("device_id", "unknown_device"),
            "distance": float(latest.get("distance", -1.0)),
            "is_apriltag_present": bool(latest.get("is_apriltag_present", False)),
            "apriltag_center": latest.get("apriltag_center", [-1, -1]),
            "timestamp": latest.get("timestamp", time.time()),
        },
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
