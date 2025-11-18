import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from distance.estimate_dist import calc_dist

OG_W = 320
OG_H = 240 
IMG_W = 64 
IMG_H = 64

def plot_predicted_corners(img_array, corners, label="pred"):
    plt.figure(figsize=(5,5))
    if img_array.ndim == 2:
        plt.imshow(img_array, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    x_coords = corners[0::2]
    y_coords = corners[1::2]

    for j in range(4):
        curr_x = [x_coords[j], x_coords[(j + 1) % 4]]
        curr_y = [y_coords[j], y_coords[(j + 1) % 4]]
        plt.plot(curr_x, curr_y, c="red", linewidth=2)

    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    plt.scatter(center_x, center_y, c="yellow", s=40)
    plt.text(center_x, center_y, label, color="yellow", fontsize=12, ha="center")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


model = load_model("apriltag_regressor_finetuned.keras")

img_path = input("Input path: ").strip()
og_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if og_img is None:
    raise ValueError(f"Image not found: {img_path}")
img_resized = cv2.resize(og_img, (IMG_W, IMG_H))
img_resized = img_resized.astype(np.float32) / 255.0
img_input = np.expand_dims(img_resized, axis=(0, -1))  # shape: (1,64,64,1)

corner_pred = model.predict(img_input)[0]  # shape: (8,)
pred_corners_px = corner_pred.copy()
pred_corners_px[0::2] *= OG_W
pred_corners_px[1::2] *= OG_H

print("Predicted corner coordinates (pixels):")
print(pred_corners_px.reshape(4, 2))
calc_dist(pred_corners_px.reshape(4, 2))
plot_predicted_corners(og_img, pred_corners_px)
