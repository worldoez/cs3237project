import tensorflow as tf
import numpy as np
import cv2

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="apriltag_multitask_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Pick a sample image to test
img = cv2.imread(input("Input img path: "), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
img = np.expand_dims(img, axis=(0, -1))

# Convert to quantized dtype if needed
if input_details[0]["dtype"] == np.uint8:
    scale, zero_point = input_details[0]['quantization']
    img = (img / scale + zero_point).astype(np.uint8)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Dequantize outputs if needed
for i, out in enumerate(output_details):
    if out["dtype"] == np.uint8:
        scale, zero_point = out["quantization"]
        y = interpreter.get_tensor(out['index'])
        y = (y.astype(np.float32) - zero_point) * scale
        print(f"Output {i} (dequantized):", y)
    else:
        y = interpreter.get_tensor(out['index'])
        print(f"Output {i}:", y)

reg_output = interpreter.get_tensor(output_details[0]['index'])  # 8 floats
cls_output = interpreter.get_tensor(output_details[1]['index'])  # 1 float

print("Classification prob:", float(cls_output))
print("Predicted corners (normalized):", reg_output.reshape(-1))
