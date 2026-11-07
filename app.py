import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.inference import predict_image, MODEL_VERSION
from src.database import initialize_database, save_prediction, get_history

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

initialize_database()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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
                        result = predict_image(file.stream)
                        save_prediction(filename, result)
                    except Exception:
                        result = None
                        error = (
                            "Could not process this image. "
                            "Please upload a valid PNG, JPG, or WEBP image."
                        )

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
    return {
        "status": "healthy",
        "model": "VisionGuardClassifier",
        "model_version": str(MODEL_VERSION)
    }, 200


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result=None,
        error=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        model_version=MODEL_VERSION
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
