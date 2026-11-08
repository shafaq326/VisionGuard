import os
import requests

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.database import initialize_database, save_prediction, get_history

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

MODEL_API_URL = os.getenv("MODEL_API_URL", "").rstrip("/")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

initialize_database()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_using_cloud_api(image_file, filename):
    if not MODEL_API_URL:
        raise RuntimeError("Cloud model API URL is not configured.")

    response = requests.post(
        f"{MODEL_API_URL}/predict",
        files={
            "file": (
                filename,
                image_file,
                getattr(image_file, "mimetype", "application/octet-stream")
            )
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Please select an image."
        else:
            file = request.files["image"]

            if file.filename == "":
                error = "No image selected."
            elif not allowed_file(file.filename):
                error = "Only PNG, JPG, JPEG, and WEBP files are allowed."
            else:
                filename = secure_filename(file.filename)

                if not filename:
                    error = "Invalid filename."
                else:
                    try:
                        result = predict_using_cloud_api(file.stream, filename)
                        save_prediction(filename, result)

                    except requests.exceptions.RequestException as e:
                        result = None
                        error = f"Cloud API request failed: {str(e)}"

                    except Exception as e:
                        result = None
                        error = f"Could not process image: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        model_version=MODEL_VERSION
    )


@app.route("/history", methods=["GET"])
def history():
    records = get_history()
    return render_template("history.html", records=records)


@app.route("/health", methods=["GET"])
def health():
    try:
        response = requests.get(
            f"{MODEL_API_URL}/health",
            timeout=15
        )
        response.raise_for_status()
        return response.json(), 200

    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e)
        }, 503


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result=None,
        error=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        model_version=MODEL_VERSION
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
