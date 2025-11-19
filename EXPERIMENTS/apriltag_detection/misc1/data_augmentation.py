import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
import os

AUGMENTED_OUTPUT_CSV = "augmented_apriltag_train_data.csv"
AUGMENTED_OUTPUT_DIR = "captured_frames"

def read_image_for_augmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    return img 

augment_with_corners = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-5, 5),
        mode=cv2.BORDER_REFLECT_101,
        p=0.9
    )
],
keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

augment_no_corners = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(p=0.4),
    A.MotionBlur(p=0.2),
    A.ISONoise(p=0.15)
], p=1.0)

def augment_and_save_images(df_ranged, output_dir=AUGMENTED_OUTPUT_DIR, num_augs=4):
    os.makedirs(output_dir, exist_ok=True)
    new_rows = []
    counter = 0

    for _, row in df_ranged.iterrows():
        img_filepath = row["img_filepath"]
        has_apriltag = bool(row["has_apriltag"])

        try:
            img = read_image_for_augmentation(img_filepath)
        except FileNotFoundError:
            print(f"Skipping missing: {img_filepath}")
            continue

        if has_apriltag:
            corners_col = row["corners"]
            if not isinstance(corners_col, str) or corners_col.strip() == "":
                continue

            corners = np.array([float(x) for x in corners_col.split(",")], dtype=np.float32)
            c = corners.reshape(-1, 2)

            for _ in range(num_augs):
                augmented = augment_with_corners(image=img, keypoints=c)
                aug_img = augmented["image"]
                aug_corners = np.array(augmented["keypoints"])

                new_filename = f"frame_{10000000 + counter}.jpg"
                save_path = os.path.join(output_dir, new_filename)

                cv2.imwrite(save_path, aug_img)
                counter += 1

                new_rows.append({
                    "img_filepath": save_path,
                    "has_apriltag": True,
                    "corners": ",".join(map(str, aug_corners.flatten().astype(np.float32)))
                })

                print(f"Saved: {save_path} (with AprilTag)")

        else:
            for _ in range(num_augs):
                augmented = augment_no_corners(image=img)
                aug_img = augmented["image"]

                new_filename = f"frame_{10000000 + counter}.jpg"
                save_path = os.path.join(output_dir, new_filename)

                cv2.imwrite(save_path, aug_img)
                counter += 1

                new_rows.append({
                    "img_filepath": save_path,
                    "has_apriltag": False,
                    "corners": ""
                })

                print(f"Saved: {save_path} (no AprilTag)")

    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv("augmented_apriltag_train_data.csv", index=False)
    print(f"Saved {len(new_rows)} augmented images and metadata to augmented output csv")
    return augmented_df

def show_augmented_example(img, corners):
    augmented = augment_with_corners(image=img, keypoints=corners.reshape(-1, 2))
    aug_img = augmented["image"]
    aug_corners = np.array(augmented["keypoints"])

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    plt.scatter(aug_corners[:, 0], aug_corners[:, 1], color="red", s=30)
    plt.title("Example Augmentation")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("apriltag_train_data.csv")
    #df_ranged = df.iloc[2811:2812]
    augmented_df = augment_and_save_images(df)
