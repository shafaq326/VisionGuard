
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mlflow.tracking import MlflowClient
import mlflow.artifacts

IMAGE_SIZE = (160, 160)
THRESHOLD = 0.265
MODEL_NAME = "VisionGuardClassifier"
ALIAS = "champion"


def load_champion_model():
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    artifact_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=champion.source
    )

    artifact_path = Path(artifact_dir)

    if (artifact_path / "model.keras").exists():
        model_path = artifact_path / "model.keras"
    elif (artifact_path / "saved_model.pb").exists():
        model_path = artifact_path
    else:
        keras_files = list(artifact_path.glob("*.keras"))
        if not keras_files:
            raise FileNotFoundError("Champion model file not found.")
        model_path = keras_files[0]

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input}
    )

    return model, champion.version


MODEL, MODEL_VERSION = load_champion_model()


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability_real = float(MODEL.predict(image_array, verbose=0)[0][0])

    prediction = (
        "Real" if probability_real >= THRESHOLD else "AI-generated"
    )

    return {
        "prediction": prediction,
        "probability_real": probability_real,
        "probability_ai": 1.0 - probability_real,
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION
    }
