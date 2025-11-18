import pandas as pd
import numpy as np
import cv2

class AprilTag:
    def __init__(self, corners_arr=None, img_filepath=None, csv_path=None, csv_row=None):
        # Attributes: bottom_left, bottom_right, top_right, top_left, original_length, img_filepath
        if csv_path and csv_row:
            df = pd.read_csv(csv_path)
            self.img_filepath = df['img_filepath']
            corners_arr = df['corners'].reshape(4, 2)
        if corners_arr and img_filepath:
            self.img_filepath = img_filepath
            #self.corners_arr = corners_arr
            self.original_length = 100 # measured in mm
            
        center = np.mean(corners_arr, axis=0) # to save space, using computed centre instead
        angles = np.arctan2(corners_arr[:,1] - center[1], corners_arr[:,0] - center[0])
        print(angles)
        sort_idx = np.argsort(angles)
            
        self.bottom_left, self.bottom_right, self.top_right, self.top_left = corners_arr[sort_idx]
    
    def __str__(self):
        print("Tag ID:", self.tag_id)
        print("Centre:", self.center)
        print("Top Left:", self.top_left)
        print("Top Right:", self.top_right)
        print("Bottom Left:", self.bottom_left)
        print("Bottom Right:", self.bottom_right)
        
    def reorder_corners(corners):
        # Reorder corners arr in TL TR BR BL order
        corners = np.array(corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(f"Expected 4 corners of shape (4,2), got {corners.shape}")
        s = corners.sum(axis=1) # x + y -> TL has smallest sum, BR has largest
        diff = np.diff(corners, axis=1)  # y - x -> TR smallest diff, BL largest

        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]

        ordered = np.array([tl, tr, br, bl], dtype=np.float32)
        return ordered
