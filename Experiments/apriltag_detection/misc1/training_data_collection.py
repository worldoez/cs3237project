import cv2
import requests
import numpy as np
import os

url = "http://192.168.4.1:81/stream"
stream = requests.get(url, stream=True)

bytes_data = b''
frame_count = 0

for chunk in stream.iter_content(chunk_size=1024):
    print("chunk", chunk)
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        print("Parsing...")
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        curr_frames_len = len(list(os.scandir("captured_frames")))
        if img is not None:
            cv2.imshow('ESP32 Stream', img)
            cv2.imwrite(f"captured_frames/frame_{curr_frames_len+frame_count+1}.jpg", img)
            print(f"Saved frame_{curr_frames_len+frame_count+1}.jpg")
            frame_count += 1

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
print("Done.")
