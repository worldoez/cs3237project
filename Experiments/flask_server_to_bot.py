from flask import Flask, jsonify, request
import psycopg2
from psycopg2 import pool
import requests
from contextlib import contextmanager
import time, threading

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
    DIST_URL = "http://192.168.4.6:5001/getDistance"   # camera
    IMU_URL  = "http://192.168.4.6:5000/control"       # imu server
    timeout = 1.2

    distance = None
    distance_ts = None
    command = None
    command_ts = None

    # 1) get distance (fault-tolerant)
    try:
        r_dist = requests.get(DIST_URL, timeout=timeout)
        r_dist.raise_for_status()
        j = r_dist.json()
        # adapt to the other server's payload shape
        if isinstance(j, dict) and "data" in j and "distance" in j["data"]:
            distance = float(j["data"]["distance"])
            distance_ts = float(j["data"].get("timestamp", time.time()))
        else:
            # try alternate shape
            distance = float(j.get("distance", -1))
            distance_ts = float(j.get("timestamp", time.time()))
    except Exception as e:
        distance = -1
        distance_ts = time.time()

    # 2) get imu control (fault-tolerant)
    try:
        r_imu = requests.get(IMU_URL, timeout=timeout)
        r_imu.raise_for_status()
        j = r_imu.json()
        command = j.get("command", None)
        # server may or may not include timestamp; prefer returned 'timestamp' if present
        command_ts = float(j.get("timestamp", time.time()))
    except Exception as e:
        command = None
        command_ts = time.time()


    # try:
    #     responseDist = requests.get("http://192.168.4.6:5001/getDistance", timeout=5)
    #     responseImu = requests.get("http://192.168.4.6:5000/control", timeout=5)

    #     # print(responseDist.json())
    #     print(responseImu.json())
    #     responseDist.raise_for_status()
    #     responseImu.raise_for_status()

    #     distance = float(responseDist.json()["data"]["distance"])
    #     command = responseImu.json()["command"]

        # with get_db_connection() as conn:
        #     with conn.cursor() as cur:
        #         cur.execute(
        #             "INSERT INTO test_table (distance, command) VALUES (%s, %s)",
        #             (distance, command),
        #         )

        # 9 too far >= 55
        # 10 too near <= 22
        print(distance)

        if command == "JUMP":
            commandNum = 8
        elif distance == -1:
            commandNum = 0
        elif distance >= 55:
            commandNum = 9
        # elif distance <= 25:
        elif distance is not None and distance <= 25 and distance >= 0:
            commandNum = 10
        elif command == "STRAIGHT":
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
        elif command == "JUMP":
            commandNum = 8
        else:
            commandNum = 0

        # return str(commandNum)

        # 3) Insert to DB asynchronously so response isn't blocked by DB slowness
        def _db_insert_async(dist, dist_ts, cmd, cmd_ts, cmdnum):
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO test_table (distance, distance_ts, command, command_ts, command_num) VALUES (%s,%s,%s,%s,%s)",
                            (dist, dist_ts, cmd, cmd_ts, cmdnum),
                        )
            except Exception:
                pass

        threading.Thread(target=_db_insert_async, args=(distance, distance_ts, command, command_ts, commandNum), daemon=True).start()

        return jsonify({
        "ok": True,
        "distance": distance,
        "distance_ts": distance_ts,
        "command": command,
        "command_ts": command_ts,
        "command_num": commandNum
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
