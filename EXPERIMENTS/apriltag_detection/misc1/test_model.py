import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def plot_predicted_corners(img_array, corners, label="pred"):
    plt.figure()
    if img_array.ndim == 2:
        plt.imshow(img_array, cmap="gray")
    else:
        plt.imshow(img_array)

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

# Load the trained model
model = load_model("apriltag_multitask.keras")

# Load your image
img_path = input("Input path: ")
og_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
if og_img is None:
    raise ValueError("Image not found!")

# Resize to 64x64
img = cv2.resize(og_img, (64, 64))

# Normalize
img = img.astype(np.float32) / 255.0

img_array = np.expand_dims(img, axis=(0, -1))  # shape: (1, 64, 64, 1)

cls_pred, corner_pred = model.predict(img_array)

cls_label = int(cls_pred[0][0] > 0.5)
print("Has AprilTag?" , cls_label)

predicted_corners = corner_pred[0] * 240
print("Predicted corners:", predicted_corners)
plot_predicted_corners(og_img, predicted_corners)