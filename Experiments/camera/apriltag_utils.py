from pupil_apriltags import Detector
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def initialise_detector():
    return Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=0.8,  # less downscaling gives more detail (default is 2.0)
        quad_sigma=0.2,  # smooth noise so quads form more easily
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

def is_apriltag_present(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[0, :, :, 0]
    detector = initialise_detector()
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    detection = detector.detect(img_array)

    return len(detection) > 0


def compute_centre_from_img(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[0, :, :, 0]
    detector = initialise_detector()
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    detection = detector.detect(img_array)

    return detection.center

def compute_corners_from_img(img_array):
    img_array = (img_array * 255).astype(np.uint8)
    img_array = img_array[0, :, :, 0]
    detector = initialise_detector()
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    detection = detector.detect(img_array)
    print("detection:", detection)

    all_corners = []

    for det in detection:
        corners = det.corners
        all_corners.append(corners)

    return np.array(all_corners)

def detect_apriltag_from_array(img_array, detector, is_plot=True):
    img_array = cv2.cvtColor(
        img_array, cv2.COLOR_RGBA2GRAY
    )  # change rgba to black and white channels if any
    # print("Img array shape:", img_array.shape)

    detection = detector.detect(img_array)
    # print(len(detection), "april tags found")

    for i in range(len(detection)):
        april_tag_detected = detection[i]
        # print(i, april_tag_detected)
        corners = detection[i].corners
        x_coords = []
        y_coords = []
        for corner in corners:
            x_coords.append(corner[0])
            y_coords.append(corner[1])

        if is_plot:
            center = detection[i].center
            plt.text(
                center[0],
                center[1],
                str(detection[i].tag_id),
                color="yellow",
                fontsize=12,
                ha="center",
            )

        for j in range(4):
            curr_x_coord = [x_coords[j], x_coords[(j + 1) % 4]]
            curr_y_coord = [y_coords[j], y_coords[(j + 1) % 4]]
            if is_plot:
                plt.plot(curr_x_coord, curr_y_coord, c="red")

    if is_plot:
        plt.tight_layout()
        plt.show()

    return detection

def detect_apriltag_from_image(img_filepath, detector, is_plot=True):
    image = Image.open(img_filepath)
    if is_plot:
        plt.imshow(image)
    img_array = np.asarray(image, dtype=np.uint8)
    detection = detect_apriltag_from_array(img_array, detector, is_plot=is_plot)
    return detection

def plot_predicted_corners(img_array, corners, label="Predicted"):
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


if __name__ == "__main__":
    detector = initialise_detector()
    filepath = "captured_frames/frame_5629.jpg"
    detection = detect_apriltag_from_image(filepath, detector)
    # print(detection)
