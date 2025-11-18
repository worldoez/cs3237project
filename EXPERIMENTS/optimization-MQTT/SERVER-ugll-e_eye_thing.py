# import numpy as np
# import cv2
# import paho.mqtt.client as mqtt
# import json
# import base64
# from tensorflow.keras.models import load_model
# from pupil_apriltags import Detector
# import time

# # ===========================
# # AprilTag Detection Functions
# # ===========================
# def initialise_detector():
#     return Detector(
#         families="tag36h11",
#         nthreads=1,
#         quad_decimate=0.8,
#         quad_sigma=0.2,
#         refine_edges=1,
#         decode_sharpening=0.25,
#         debug=0
#     )

# def is_apriltag_present(img_array):
#     img_array = (img_array * 255).astype(np.uint8)
#     img_array = img_array[0, :, :, 0]
#     detector = initialise_detector()
#     detection = detector.detect(img_array)
#     return len(detection) > 0

# def compute_corners_from_img(img_array):
#     img_array = (img_array * 255).astype(np.uint8)
#     img_array = img_array[0, :, :, 0]
#     detector = initialise_detector()
#     detection = detector.detect(img_array)
    
#     all_corners = []
#     for det in detection:
#         corners = det.corners
#         all_corners.append(corners)
    
#     return np.array(all_corners)

# # ===========================
# # Distance Calculation Functions
# # ===========================
# def calc_dist(corners):
#     """Calculate distance from AprilTag corners using PnP"""
#     image_points = np.array(
#         [
#             [corners[0]],  # top-left
#             [corners[1]],  # top-right
#             [corners[2]],  # bottom-right
#             [corners[3]],  # bottom-left
#         ],
#         dtype=np.float32,
#     )
    
#     if np.mean(image_points) < 10:
#         return -1
    
#     S = 0.10  # tag side in meters
#     object_points = np.array(
#         [
#             [-S / 2, S / 2, 0.0],
#             [S / 2, S / 2, 0.0],
#             [S / 2, -S / 2, 0.0],
#             [-S / 2, -S / 2, 0.0],
#         ],
#         dtype=np.float32,
#     )
    
#     # Camera intrinsics
#     K = np.array([
#         [240, 0, 120], 
#         [0, 240, 120], 
#         [0, 0, 1]
#     ], dtype=np.float64)
#     dist = np.zeros(5)
    
#     # Solve PnP
#     success, rvec, tvec = cv2.solvePnP(
#         object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
#     )
    
#     if not success:
#         return -1
    
#     tvec = tvec.reshape(3)
#     forward_distance = tvec[2]  # distance along camera Z-axis
#     return forward_distance * 100  # return in cm

# def compute_center_from_corners(corners_arr):
#     """Compute center point from corners"""
#     sum_coords = np.sum(corners_arr, axis=0)
#     return sum_coords / 4.0

# # ===========================
# # Image Processing Functions
# # ===========================
# def preprocess_image(img):
#     """Preprocess image for CNN model"""
#     IMG_W, IMG_H = 64, 64
#     img_resized = cv2.resize(img, (IMG_W, IMG_H))
#     img_resized = img_resized.astype(np.float32) / 255.0
#     img_input = np.expand_dims(img_resized, (0, -1))
#     return img_input

# # ===========================
# # MQTT Camera Server Class
# # ===========================
# class MQTTCameraServer:
#     def __init__(self, broker="localhost", port=1883):
#         self.broker = broker
#         self.port = port
#         self.client = mqtt.Client(client_id="camera_server", protocol=mqtt.MQTTv311)
#         self.OG_W, self.OG_H = 240, 240
#         self.corners_model = None
#         self.last_process_time = 0
#         self.interval_time = 0.1
        
#         # MQTT callbacks
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
        
#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             print(f"Connected to MQTT broker at {self.broker}:{self.port}")
#             # Subscribe to camera topic
#             client.subscribe("/cam")
#             print("Subscribed to /cam topic")
#         else:
#             print(f"Failed to connect, return code {rc}")
    
#     def on_message(self, client, userdata, msg):
#         """Handle incoming camera frames"""
#         try:
#             curr_time = time.time()
            
#             # Rate limiting
#             if curr_time - self.last_process_time < self.interval_time:
#                 return
            
#             self.last_process_time = curr_time
            
#             # Decode the message
#             payload = json.loads(msg.payload.decode())
            
#             # Extract and decode image
#             img_base64 = payload.get("image", "")
#             img_bytes = base64.b64decode(img_base64)
#             img_array = np.frombuffer(img_bytes, dtype=np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
#             if img is None:
#                 print("[ERROR] Failed to decode image")
#                 return
            
#             # Process the image
#             self.process_frame(img, payload.get("device_id", "unknown"))
            
#         except Exception as e:
#             print(f"[ERROR] Processing message: {e}")
    
#     def process_frame(self, img, device_id):
#         """Process a single frame and detect AprilTags"""
#         try:
#             # Display the frame
#             cv2.imshow('ESP32 MQTT Stream', img)
#             cv2.waitKey(1)
            
#             # Preprocess for detection
#             img_input = preprocess_image(img)
#             computed_center = [-1, -1]
            
#             # Check for AprilTag presence
#             if not is_apriltag_present(img_input):
#                 print("No AprilTags detected")
#                 forward_distance = -1.0
#                 is_tag_present = False
#             else:
#                 # Detect corners
#                 corner_pred = compute_corners_from_img(img_input)
#                 computed_center = compute_center_from_corners(corner_pred)
                
#                 # Scale corners to original image size
#                 pred_corners_px = corner_pred.copy()
#                 pred_corners_px[0::2] *= self.OG_W
#                 pred_corners_px[1::2] *= self.OG_H
#                 corners = pred_corners_px.reshape(4, 2)
                
#                 print(f"Detected corners: {corners}")
                
#                 # Calculate distance
#                 forward_distance = calc_dist(corners)
#                 print(f"[INFO] Distance: {forward_distance:.3f} cm")
#                 is_tag_present = True
            
#             # Prepare response payload
#             response = {
#                 "device_id": device_id,
#                 "distance": float(forward_distance),
#                 "is_apriltag_present": is_tag_present,
#                 "apriltag_center": computed_center.tolist() if isinstance(computed_center, np.ndarray) else computed_center,
#                 "timestamp": time.time(),
#             }
            
#             # Publish to /cam_out
#             self.client.publish("/cam_out", json.dumps(response), qos=1)
#             print(f"Published to /cam_out: distance={forward_distance:.2f} cm")
            
#         except Exception as e:
#             print(f"[ERROR] Processing frame: {e}")
    
#     def start(self):
#         """Connect to MQTT broker and start listening"""
#         print(f"Connecting to MQTT broker {self.broker}:{self.port}...")
#         self.client.connect(self.broker, self.port, 60)
        
#         print("MQTT Camera Server started. Waiting for camera frames on /cam...")
#         print("Press Ctrl+C to stop")
        
#         try:
#             self.client.loop_forever()
#         except KeyboardInterrupt:
#             print("\nStopping MQTT Camera Server...")
#         finally:
#             cv2.destroyAllWindows()
#             self.client.disconnect()
#             print("Server stopped")

# # ===========================
# # Main
# # ===========================
# if __name__ == "__main__":
#     # Configuration
#     BROKER = "localhost"  # Change to your MQTT broker address
#     PORT = 1883
    
#     # Create and start server
#     server = MQTTCameraServer(broker=BROKER, port=PORT)
#     server.start()


import numpy as np
import cv2
import paho.mqtt.client as mqtt
import json
import base64
from tensorflow.keras.models import load_model
from pupil_apriltags import Detector
import time
import argparse

# ===========================
# AprilTag Detection Functions
# ===========================
def initialise_detector():
    return Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=0.8,
        quad_sigma=0.2,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

def is_apriltag_present(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[0, :, :, 0]
    detector = initialise_detector()
    detection = detector.detect(img_array)
    return len(detection) > 0

def compute_corners_from_img(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[0, :, :, 0]
    detector = initialise_detector()
    detection = detector.detect(img_array)
    
    all_corners = []
    for det in detection:
        corners = det.corners
        all_corners.append(corners)
    
    return np.array(all_corners)

# ===========================
# Distance Calculation Functions
# ===========================
def calc_dist(corners):
    """Calculate distance from AprilTag corners using PnP"""
    image_points = np.array(
        [
            [corners[0]],  # top-left
            [corners[1]],  # top-right
            [corners[2]],  # bottom-right
            [corners[3]],  # bottom-left
        ],
        dtype=np.float32,
    )
    
    if np.mean(image_points) < 10:
        return -1
    
    S = 0.10  # tag side in meters
    object_points = np.array(
        [
            [-S / 2, S / 2, 0.0],
            [S / 2, S / 2, 0.0],
            [S / 2, -S / 2, 0.0],
            [-S / 2, -S / 2, 0.0],
        ],
        dtype=np.float32,
    )
    
    # Camera intrinsics
    K = np.array([
        [240, 0, 120], 
        [0, 240, 120], 
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.zeros(5)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return -1
    
    tvec = tvec.reshape(3)
    forward_distance = tvec[2]  # distance along camera Z-axis
    return forward_distance * 100  # return in cm

def compute_center_from_corners(corners_arr):
    """Compute center point from corners"""
    sum_coords = np.sum(corners_arr, axis=0)
    return sum_coords / 4.0

# ===========================
# Image Processing Functions
# ===========================
def preprocess_image(img):
    """Preprocess image for CNN model"""
    IMG_W, IMG_H = 64, 64
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, (0, -1))
    return img_input

# ===========================
# MQTT Camera Server Class
# ===========================
class MQTTCameraServer:
    def __init__(self, broker="localhost", port=1883, interval=0.1):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client(client_id="camera_server", protocol=mqtt.MQTTv311)
        self.OG_W, self.OG_H = 240, 240
        self.corners_model = None
        self.last_process_time = 0
        self.interval_time = interval
        
        # MQTT callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {self.broker}:{self.port}")
            # Subscribe to camera topic
            client.subscribe("/cam")
            print("Subscribed to /cam topic")
        else:
            print(f"Failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming camera frames"""
        try:
            curr_time = time.time()
            
            # Rate limiting
            if curr_time - self.last_process_time < self.interval_time:
                return
            
            self.last_process_time = curr_time
            
            # Decode the message
            payload = json.loads(msg.payload.decode())
            
            # Extract and decode image
            img_base64 = payload.get("image", "")
            img_bytes = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print("[ERROR] Failed to decode image")
                return
            
            # Process the image
            self.process_frame(img, payload.get("device_id", "unknown"))
            
        except Exception as e:
            print(f"[ERROR] Processing message: {e}")
    
    def process_frame(self, img, device_id):
        """Process a single frame and detect AprilTags"""
        try:
            # Display the frame
            cv2.imshow('ESP32 MQTT Stream', img)
            cv2.waitKey(1)
            
            # Preprocess for detection
            img_input = preprocess_image(img)
            computed_center = [-1, -1]
            
            # Check for AprilTag presence
            if not is_apriltag_present(img_input):
                print("No AprilTags detected")
                forward_distance = -1.0
                is_tag_present = False
            else:
                # Detect corners
                corner_pred = compute_corners_from_img(img_input)
                computed_center = compute_center_from_corners(corner_pred)
                
                # Scale corners to original image size
                pred_corners_px = corner_pred.copy()
                pred_corners_px[0::2] *= self.OG_W
                pred_corners_px[1::2] *= self.OG_H
                corners = pred_corners_px.reshape(4, 2)
                
                print(f"Detected corners: {corners}")
                
                # Calculate distance
                forward_distance = calc_dist(corners)
                print(f"[INFO] Distance: {forward_distance:.3f} cm")
                is_tag_present = True
            
            # Prepare response payload
            response = {
                "device_id": device_id,
                "distance": float(forward_distance),
                "is_apriltag_present": is_tag_present,
                "apriltag_center": computed_center.tolist() if isinstance(computed_center, np.ndarray) else computed_center,
                "timestamp": time.time(),
            }
            
            # Publish to /cam_out
            self.client.publish("/cam_out", json.dumps(response), qos=1)
            print(f"Published to /cam_out: distance={forward_distance:.2f} cm")
            
        except Exception as e:
            print(f"[ERROR] Processing frame: {e}")
    
    def start(self):
        """Connect to MQTT broker and start listening"""
        print(f"Connecting to MQTT broker {self.broker}:{self.port}...")
        self.client.connect(self.broker, self.port, 60)
        
        print("MQTT Camera Server started. Waiting for camera frames on /cam...")
        print("Press Ctrl+C to stop")
        
        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\nStopping MQTT Camera Server...")
        finally:
            cv2.destroyAllWindows()
            self.client.disconnect()
            print("Server stopped")

# ===========================
# Main
# ===========================
def main():
    parser = argparse.ArgumentParser(description='MQTT Camera Server with AprilTag Detection')
    
    # MQTT arguments
    parser.add_argument('--broker', '-b', default='localhost',
                        help='MQTT broker IP address (default: localhost)')
    parser.add_argument('--port', '-p', type=int, default=1883,
                        help='MQTT broker port (default: 1883)')
    
    # Processing arguments
    parser.add_argument('--interval', '-i', type=float, default=0.1,
                        help='Minimum time interval between frame processing in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("MQTT Camera Server Starting")
    print("=" * 50)
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Processing Interval: {args.interval}s")
    print("=" * 50)
    
    # Create and start server
    server = MQTTCameraServer(broker=args.broker, port=args.port, interval=args.interval)
    server.start()

if __name__ == "__main__":
    main()