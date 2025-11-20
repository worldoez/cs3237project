from apriltag_cv_detection import *
from ...Experiments.server.estimate_dist import calc_dist

import os
import pandas as pd

# if __name__ == "__main__":
#     detector = initialise_detector()
#     cap = cv2.VideoCapture(0)
#     if cap.isOpened():
#         detect_apriltag_from_cv(cap, detector)

if __name__ == "__main__":
    mode = input("mode [i]nput or mode [c]onvert: ")
    detector = initialise_detector()
    if mode == 'i':
        img_filepath = input("Input path: ")
        # image_folder_path = "captured_frames"
        # img_filepath = os.path.join(image_folder_path, "frame_52.jpg")
        detection = detect_apriltag_from_image(img_filepath, detector)
        has_apriltag = len(detection) != 0

        print("Image filepath:", img_filepath)
        print("has apriltag:", has_apriltag)
        corners = detection[0].corners.flatten()
        print("corners:", corners.flatten())
        print(calc_dist(detection[0].corners))
    else:
        image_folder_path = "captured_frames"
        data = generate_training_data(detector, image_folder_path)
        # print percentage of images with detectable apriltags
        data = pd.read_csv("apriltag_train_data.csv")
        apriltag_count = len(data[data["has_apriltag"]])
        total_rows = len(data)
        print("April tags detected:", apriltag_count)
        print(f"April tag percentage: {(100 * apriltag_count / total_rows):.2f}%")
