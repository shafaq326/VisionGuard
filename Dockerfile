FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/serving_model.keras \
    MODEL_VERSION=2 \
    MODEL_THRESHOLD=0.265 \
    MAX_UPLOAD_MB=10 \
    DATABASE_PATH=/app/instance/visionguard.db

WORKDIR /app

# Install the system library required by TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root application user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/instance

# Copy application files and serving model
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/serving_model.keras ./models/serving_model.keras

# Give the application user access to its files
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Check whether the Flask application is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
