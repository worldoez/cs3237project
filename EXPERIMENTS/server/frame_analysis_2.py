import cv2
import requests
import numpy as np
import time
from tensorflow.keras.models import load_model

OG_W, OG_H = 240, 240
IMG_W, IMG_H = 64, 64

def preprocess_image(img):
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, (0, -1))
    return img_input  # (1, 64, 64, 1)

url = "http://192.168.4.1:81/stream"
stream = requests.get(url, stream=True)
is_present_model = load_model("apriltag_binary_classifier.keras")

bytes_data = b''

last_check_time = time.time()
detections_in_window = []

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1 and a < b:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            cv2.imshow('ESP32 Stream', img)
            img_input = preprocess_image(img)

            pred = is_present_model.predict(img_input, verbose=0)[0][0]
            detections_in_window.append(pred)

            current_time = time.time()
            if current_time - last_check_time >= 1.0:
                max_pred = max(p for p in detections_in_window)
                print(max_pred)
                if max_pred > 0.3:
                    print("AprilTag detected in last second")
                else:
                    print("No AprilTag detected")

                detections_in_window.clear()
                last_check_time = current_time

        if cv2.waitKey(1) == 27: 
            break

cv2.destroyAllWindows()
stream.close()
print("Done.")
