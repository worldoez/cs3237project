import cv2
import pandas as pd
import numpy as np
import os
import time

CSV_PATH = "apriltag_train_data.csv"
SAVE_EVERY = 1  # save progress after every one labelled images
MAX_SCREEN_WIDTH = 3840
MAX_SCREEN_HEIGHT = 2160

clicked_points = []
current_image_path = None
current_index = 0
df = None
next_image_flag = False
scale_factor = 1.0  # used to map scaled clicks back to og coords

def resize_to_screen(img, max_width=MAX_SCREEN_WIDTH, max_height=MAX_SCREEN_HEIGHT):
    """Resize image to fit screen while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def redraw_points(img, points):
    img_display = img.copy()
    for i, pt in enumerate(points):
        cv2.circle(img_display, pt, 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(i + 1), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for i in range(1, len(points)):
        cv2.line(img_display, points[i - 1], points[i], (0, 255, 0), 2)
    return img_display


def click_event(event, x, y, flags, param):
    global clicked_points, current_image_path, current_index, df, next_image_flag, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: ({x}, {y})")

        img_display = redraw_points(param, clicked_points)
        cv2.imshow("Label Image", img_display)

        if len(clicked_points) == 4:
            # Convert back to original coordinates
            unscaled = [(px / scale_factor, py / scale_factor) for (px, py) in clicked_points]
            corners_str = ",".join([f"{x:.6f},{y:.6f}" for (x, y) in unscaled])
            df.at[current_index, "has_apriltag"] = True
            df.at[current_index, "corners"] = corners_str
            print(f"Saved corners for {current_image_path}: {corners_str}")
            clicked_points = []
            next_image_flag = True
            cv2.destroyAllWindows()

def save_progress():
    df.to_csv(CSV_PATH, index=False)
    print(f"saved to {CSV_PATH}")

def main(start_row=None, end_row=None):
    global clicked_points, current_image_path, current_index, df, next_image_flag, scale_factor
    newly_labelled = []
    df = pd.read_csv(CSV_PATH)

    start_idx = start_row if start_row is not None else 0
    end_idx = end_row if end_row is not None else len(df)
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        current_index = idx
        img_path = row["img_filepath"]
        has_apriltag = bool(row["has_apriltag"])
        corners = row["corners"]

        if has_apriltag and isinstance(corners, str) and len(corners) > 0:
            continue
        
        print("curr img path:", img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        img_resized, scale_factor = resize_to_screen(img)
        current_image_path = img_path
        clicked_points = []
        next_image_flag = False

        print("Click 4 corners of the AprilTag")
        print("[u] to undo point, [c] if no tag, [q] to quit")

        window_name = "Label Image"
        cv2.imshow(window_name, img_resized)
        cv2.setMouseCallback(window_name, click_event, img_resized)

        labelled_this = False  # track if this one became True

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                save_progress()
                cv2.destroyAllWindows()
                print(f"Newly labelled True images: {newly_labelled}")
                return newly_labelled

            elif key == ord('u'):
                if clicked_points:
                    removed = clicked_points.pop()
                    print(f"Undo and removed last point: {removed}")
                    img_display = redraw_points(img_resized, clicked_points)
                    cv2.imshow(window_name, img_display)

            elif key == ord('c'):
                print("No april tag here.")
                df.at[idx, "has_apriltag"] = False
                df.at[idx, "corners"] = ""
                cv2.destroyAllWindows()
                break

            elif next_image_flag:
                next_image_flag = False
                labelled_this = True
                break

        if labelled_this:
            newly_labelled.append(img_path)

        if idx % SAVE_EVERY == 0:
            save_progress()

    save_progress()
    print(f"Newly labelled True images: {newly_labelled}")
    return newly_labelled

if __name__ == "__main__":
    with open("Experiments/apriltag_detection/last_val.txt", 'r') as file:
        line = file.readline() 
        last_val = int(line.strip())
    print("last val:", last_val)
    newly_labelled = main(start_row=last_val)
    with open("Experiments/apriltag_detection/last_val.txt", 'w') as file:
        if len(newly_labelled) > 0:
            file.write(newly_labelled[-1].split('_')[2].split('.')[0])