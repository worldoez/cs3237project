import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('apriltag_regressor_finetuned.keras')
finetuned_model = load_model("apriltag_regressor_finetuned.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("apriltag_multitask_quant.tflite", "wb").write(tflite_model)
print("Saved quantized model to apriltag_multitask_quant.tflite")

converter_finetuned = tf.lite.TFLiteConverter.from_keras_model(finetuned_model)
converter_finetuned.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_finetuned = converter_finetuned.convert()

open("apriltag_multitask_quant_finetuned.tflite", "wb").write(tflite_model_finetuned)
print("Saved quantized model to apriltag_multitask_quant_finetuned.tflite")