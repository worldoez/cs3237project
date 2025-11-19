from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

df = pd.read_csv('apriltag_train_data.csv')
df = df[df['has_apriltag']]
IMG_W = 64
IMG_H = 64
images = []
corners_all = []

for i, row in df.iterrows():
    img_path = row['img_filepath']
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0

    c = np.array([float(x) for x in row['corners'].split(',')])
    c[0::2] /= 240.0 
    c[1::2] /= 240.0

    images.append(img)
    corners_all.append(c)

X = np.expand_dims(np.array(images), -1) 
y = np.array(corners_all)      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_reg = keras.Sequential([
    keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(8, activation='linear')
])

model_reg.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['mae']
)

history_regressor = model_reg.fit(
    X_train, y_train,
    epochs=80, batch_size=16,
    validation_split=0.1
)

test_loss, test_mae = model_reg.evaluate(X_test, y_test)
print("Test mae:", test_mae)
print("Test loss:", test_loss)

#model_reg.save("apriltag_corner_regressor.keras")
model_reg.save("apriltag_corner_regressor_2.keras")
