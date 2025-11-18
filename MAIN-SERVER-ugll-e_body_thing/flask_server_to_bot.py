from flask import Flask, jsonify
import psycopg2
from psycopg2 import pool
import requests
from contextlib import contextmanager

app = Flask(__name__)

db_pool = psycopg2.pool.SimpleConnectionPool(
    1,
    10,
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)


@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


@app.route("/fetchData", methods=["GET"])
def fetch_distance_from_other_server():
    try:
        responseDist = requests.get("http://192.168.4.4:5001/getDistance", timeout=5)
        responseImu = requests.get("http://192.168.4.4:5000/control", timeout=5)

        responseDist.raise_for_status()
        responseImu.raise_for_status()

        distance = float(responseDist.json()["data"]["distance"])
        command = responseImu.json()["command"]
        turn_angle_deg = responseImu.json()["turn_angle_deg"]
        is_april = responseDist.json()["data"]["is_apriltag_present"]
        timestamp = responseImu.json()["timestamp"]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO imu_cam_data (command, turn_angle_deg, distance, is_april, timestamp) VALUES (%s, %s, %s, %s, %s)",
                    (command, turn_angle_deg, distance, is_april, timestamp),
                )

        print(command)
        print(distance)

        if command == "JUMP":
            commandNum = 8
        elif command == "STRAIGHT" and distance != -1:
            commandNum = 1
        elif command == "BACKWARD":
            commandNum = 2
        elif command == "LEFT":
            commandNum = 3
        elif command == "RIGHT":
            commandNum = 4
        elif command == "SIDE_LEFT":
            commandNum = 5
        elif command == "SIDE_RIGHT":
            commandNum = 6
        elif command == "FULL_TURN":
            commandNum = 7
        else:
            commandNum = 0
        print(commandNum)
        return str(commandNum)

    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
