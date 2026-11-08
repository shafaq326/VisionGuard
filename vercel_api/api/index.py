import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="VisionGuard Model API")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "serving_model.onnx"
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.265"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")
MAX_FILE_SIZE = 4 * 1024 * 1024

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "VisionGuardClassifier",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="No image was uploaded.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 4 MB or smaller.")

    try:
        image = Image.open(io.BytesIO(contents))

        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise HTTPException(
                status_code=400,
                detail="Only PNG, JPEG, and WEBP images are supported."
            )

        image.verify()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((160, 160))

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability_real = float(
        session.run(None, {input_name: image_array})[0].reshape(-1)[0]
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
