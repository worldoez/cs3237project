import os
size_kb = os.path.getsize("apriltag_multitask_quant.tflite") / 1024
print(f"Model size: {size_kb:.2f} KB")