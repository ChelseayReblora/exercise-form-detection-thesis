import tensorflow as tf
from pathlib import Path

MODEL_DIR = Path(r"C:\Users\Neil\Downloads\backex\back_exercise_form_detector\models")

keras_model_path = MODEL_DIR / "gru_model.keras"
tflite_model_path = MODEL_DIR / "gru_model.tflite"
quantized_model_path = MODEL_DIR / "gru_model_quantized.tflite"

# Load the trained model
model = tf.keras.models.load_model(keras_model_path)

# STANDARD TFLITE CONVERSION

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Required for GRU/LSTM models
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("Saved standard TFLite model:", tflite_model_path)

# QUANTIZED MODEL

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

quantized_tflite_model = converter.convert()

with open(quantized_model_path, "wb") as f:
    f.write(quantized_tflite_model)

print("Saved quantized model:", quantized_model_path)