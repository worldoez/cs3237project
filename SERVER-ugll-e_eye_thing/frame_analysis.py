import cv2
import requests
import numpy as np

# from tensorflow.keras.models import load_model
from distance.estimate_dist import *
from apriltag_detection.basic_apriltag_roi_detection import (
    is_apriltag_present,
    compute_corners_from_img,
)
import time


def preprocess_image(img):
    IMG_W, IMG_H = 64, 64
    # img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, (0, -1))
    return img_input  # (1, 64, 64, 1)


def run_cnn_model(cnn_model="apriltag_regressor_finetuned.keras"):
    OG_W, OG_H = 240, 240
    # url = "http://192.168.4.1:81/stream"
    url = "http://192.168.4.1:81/stream"  # replace <CAMERA_IP> with WiFi.localIP() shown on camera serial

    print("Connecting to ESP32 stream...")
    stream = requests.get(url, stream=True)
    print("Stream connected")

    # corners_model = load_model(cnn_model)
    bytes_data = b""

    last_capture_time = 0

    try:
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b"\xff\xd8")
            b = bytes_data.find(b"\xff\xd9")

            if a != -1 and b != -1 and a < b:
                jpg = bytes_data[a : b + 2]
                bytes_data = bytes_data[b + 2 :]
                img = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )

                # curr_time = time.time()
                # if curr_time - last_capture_time >= interval_time:
                #     last_capture_time = curr_time

                if img is not None:
                    cv2.imshow('ESP32 Stream', img)

                    img_input = preprocess_image(img)
                    computed_center = [-1, -1]
                    
                    if not is_apriltag_present(img_input):
                        #print("No apriltags here")
                        forward_distance = -1.0
                        is_tag_present = False
                    else:
                        corner_pred = compute_corners_from_img(img_input)
                        computed_center = compute_center_from_corners(corner_pred)
                        #corner_pred = corners_model.predict(img_input, verbose=0)[0]
                        pred_corners_px = corner_pred.copy()
                        pred_corners_px[0::2] *= OG_W
                        pred_corners_px[1::2] *= OG_H
                        corners = pred_corners_px.reshape(4, 2)
                        print("corners:", corners)

                        forward_distance = calc_dist(corners)
                        print(f"[INFO] Distance: {forward_distance:.3f}")
                        is_tag_present = True

                    payload = {
                        "device_id": "cam01",
                        "distance": float(forward_distance),
                        "is_apriltag_present": is_tag_present,
                        "apriltag_center": computed_center.tolist() if isinstance(computed_center, np.ndarray) else computed_center,
                        "timestamp": time.time(),
                    }

                    try:
                        resp = requests.post("http://localhost:5001/distance", json=payload)
                        if not resp.ok:
                            #print("Sent distance of", forward_distance)
                        #else:
                            print("Failed to send:", resp.status_code, resp.text)
                    except Exception as e:
                        print("[ERROR] Sending distance:", e)

                if cv2.waitKey(1) == 27:
                    break

    except KeyboardInterrupt:
        print("Stopped manually")
    finally:
        cv2.destroyAllWindows()
        stream.close()
        print("Stream closed")


if __name__ == "__main__":
    run_cnn_model()
