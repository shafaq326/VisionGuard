
import os
import builtins
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

builtins.preprocess_input = preprocess_input

IMAGE_SIZE = (160, 160)
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.265"))
MODEL_NAME = os.getenv("MODEL_NAME", "VisionGuardClassifier")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_PATH = os.getenv("MODEL_PATH")


def load_champion_model():
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
            custom_objects={"preprocess_input": preprocess_input, "<lambda>": lambda image: preprocess_input(image * 255.0)}
        )

        return model, os.getenv("MODEL_VERSION", "local")

    from mlflow.tracking import MlflowClient
    import mlflow.artifacts

    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    artifact_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=champion.source
    )

    artifact_path = Path(artifact_dir)

    if (artifact_path / "data" / "model" / "saved_model.pb").exists():
        model_path = artifact_path / "data" / "model"
    elif (artifact_path / "model.keras").exists():
        model_path = artifact_path / "model.keras"
    elif (artifact_path / "saved_model.pb").exists():
        model_path = artifact_path
    else:
        keras_files = list(artifact_path.rglob("*.keras"))

        if not keras_files:
            raise FileNotFoundError("Champion model file not found.")

        model_path = keras_files[0]

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input, "<lambda>": lambda image: preprocess_input(image * 255.0)}
    )

    return model, champion.version


MODEL, MODEL_VERSION = load_champion_model()


def predict_image(image_file):
    image_file.seek(0)

    with Image.open(image_file) as image:
        image_format = image.format

        if image_format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Unsupported image format.")

        image.verify()

    image_file.seek(0)

    with Image.open(image_file) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability_real = float(
        MODEL.predict(image_array, verbose=0)[0][0]
    )

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
