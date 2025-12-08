#!/bin/bash
# Startup script for Vertex AI custom container
# Downloads model from GCS and starts the inference server

set -e

echo "Starting Bloom ML Inference Service..."

# Get GCS bucket from environment or use default
GCS_MODEL_PATH="${GCS_MODEL_PATH:-gs://bloom-health-ml-models/latest}"

echo "Downloading model from: $GCS_MODEL_PATH"

# Download model files from GCS
# Using gcloud auth from the attached service account
gsutil -m cp -r "${GCS_MODEL_PATH}/*" /model/

echo "Model downloaded. Contents:"
ls -la /model/

echo "Starting inference server on port 8080..."
exec uvicorn inference:app --host 0.0.0.0 --port 8080
