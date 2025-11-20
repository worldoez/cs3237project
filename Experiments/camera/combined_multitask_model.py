import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import Huber
import tensorflow as tf

OG_H, OG_W = 240, 240
IMG_H, IMG_W = 64, 64
df = pd.read_csv('apriltag_train_data.csv') 
images, cls_labels, corners_all = [], [], []

for _, row in df.iterrows():
    img_path = row["img_filepath"]
    has_tag = bool(row["has_apriltag"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0
    cls_labels.append(1 if has_tag else 0)

    if has_tag:
        c = np.array([float(x) for x in row["corners"].split(",")])
        c[0::2] /= OG_W
        c[1::2] /= OG_H
    else:
        c = np.zeros(8, dtype=np.float32)

    images.append(img)
    corners_all.append(c)

X = np.expand_dims(np.array(images), -1)
y_cls = np.array(cls_labels)
y_reg = np.array(corners_all)

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

inputs = keras.Input(shape=(IMG_H, IMG_W, 1))
x = layers.Conv2D(16, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

cls_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)
reg_output = layers.Dense(8, activation='linear', name='corner_output')(x)

model = models.Model(inputs=inputs, outputs=[cls_output, reg_output])

model.compile(
    optimizer='adam',
    loss={'class_output': 'binary_crossentropy', 'corner_output': 'mse'},
    loss_weights={'class_output': 1.0, 'corner_output': 5.0},
    metrics={'class_output': ['accuracy'], 'corner_output': ['mae']}
)

history = model.fit(
    X_train,
    {'class_output': y_cls_train, 'corner_output': y_reg_train},
    epochs=50,
    batch_size=16,
    validation_split=0.1
)

eval_results = model.evaluate(
    X_test, {'class_output': y_cls_test, 'corner_output': y_reg_test}
)
print("Evaluation:", eval_results)

# inspect samples at random 
def inspect_predictions(model, df, X, y_cls, y_reg, OG_W, OG_H, n=5):
    for idx in np.random.choice(len(X), n, replace=False):
        img = X[idx]
        true_cls = y_cls[idx]
        true_corners = y_reg[idx]

        pred_cls, pred_corners = model.predict(img[np.newaxis, ...])
        pred_prob = float(pred_cls.flatten()[0])
        pred_corners = pred_corners.flatten()

        pred_px = pred_corners.copy()
        pred_px[0::2] *= OG_W
        pred_px[1::2] *= OG_H

        true_px = true_corners.copy()
        true_px[0::2] *= OG_W
        true_px[1::2] *= OG_H

        mae_px = np.mean(np.abs(pred_px - true_px))
        print(f"Image {idx}: cls={true_cls}, pred_prob={pred_prob:.3f}, MAE_px={mae_px:.2f}")

        img_vis = cv2.resize((img.squeeze()*255).astype(np.uint8), (OG_W, OG_H))
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

        cv2.polylines(img_vis, [pred_px.reshape(-1,2).astype(int)], True, (0,0,255), 2)
        cv2.polylines(img_vis, [true_px.reshape(-1,2).astype(int)], True, (0,255,0), 1)

        plt.figure(figsize=(5,5))
        plt.imshow(img_vis[..., ::-1])
        plt.title("MAE_px", mae_px)
        plt.axis('off')
        plt.show()

inspect_predictions(model, df, X_test, y_cls_test, y_reg_test, OG_W, OG_H, n=5)

# Finetuning on positives only
pos_idx = np.where(y_cls_train == 1)[0]
X_pos, y_pos = X_train[pos_idx], y_reg_train[pos_idx]

reg_head = models.Model(inputs=model.input, outputs=model.get_layer('corner_output').output)
for layer in reg_head.layers[:-1]:
    layer.trainable = False

reg_head.compile(optimizer=keras.optimizers.Adam(3e-4), loss=Huber(), metrics=['mae'])
reg_head.fit(X_pos, y_pos, epochs=30, batch_size=16, validation_split=0.1)

model.save("apriltag_multitask_model.keras")
reg_head.save("apriltag_regressor_finetuned.keras")


