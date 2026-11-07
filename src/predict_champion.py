import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from mlflow.tracking import MlflowClient
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_NAME = "VisionGuardClassifier"
THRESHOLD = 0.265
IMAGE_SIZE = (160, 160)

if len(sys.argv) != 2:
    print("Usage: python src/predict_champion.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

mlflow.set_tracking_uri(
    "https://dagshub.com/shafaq_hanif/VisionGuard.mlflow"
)

client = MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

print("Champion version:", champion.version)
print("Downloading champion model...")

local_path = Path(
    mlflow.artifacts.download_artifacts(
        artifact_uri=champion.source
    )
)

keras_files = list(local_path.rglob("*.keras"))
saved_models = list(local_path.rglob("saved_model.pb"))

custom_objects = {
    "preprocess_input": preprocess_input
}

if keras_files:
    model_path = keras_files[0]
elif saved_models:
    model_path = saved_models[0].parent
else:
    raise FileNotFoundError(
        f"No Keras model found in downloaded artifacts: {local_path}"
    )

model = tf.keras.models.load_model(
    str(model_path),
    compile=False,
    safe_mode=False,
    custom_objects=custom_objects
)

image = tf.keras.utils.load_img(
    str(image_path),
    target_size=IMAGE_SIZE
)

image_array = tf.keras.utils.img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

probability_real = float(
    model.predict(image_array, verbose=0)[0][0]
)

prediction = (
    "Real"
    if probability_real >= THRESHOLD
    else "AI-generated"
)

print("\nPrediction:", prediction)
print(f"Probability of being Real: {probability_real:.4f}")
print(f"Decision threshold: {THRESHOLD}")
