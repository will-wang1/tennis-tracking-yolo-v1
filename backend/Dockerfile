# API container - deliberately lightweight: it only needs
# src/analysis/court_calibration.py (numpy + opencv-headless) for the
# calibration endpoints, not the full ML pipeline. The worker (see
# Dockerfile.worker) is the one that needs torch/tensorflow/ultralytics and
# real model weights.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY src src
COPY backend backend

WORKDIR /app/backend
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
