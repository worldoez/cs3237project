import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('apriltag_train_data.csv')
images, labels = [], []


for i, row in df.iterrows():
    img_path = row["img_filepath"]
    label = 1 if row["has_apriltag"] else 0

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    images.append(img)
    labels.append(label)

X = np.expand_dims(np.array(images), axis=-1) 
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_cls = keras.Sequential([
    keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_cls.compile(
    optimizer='adam',
    #optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    loss='binary_crossentropy',
    metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# history = model_cls.fit(
#     X_train_sub, y_train_sub,
#     validation_data=(X_val, y_val),
#     epochs=80,
#     batch_size=16,
#     callbacks=[early_stop]
# )
history = model_cls.fit(
    X_train, y_train, 
    epochs=80, 
    batch_size=16, 
    validation_split=0.1,
    callbacks=[early_stop]
)
test_loss, test_acc = model_cls.evaluate(X_test, y_test)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

print(f"Test accuracy: {test_acc:.3f}")

model_cls.save("apriltag_binary_classifier.keras")
