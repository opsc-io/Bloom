# ML Training Pipeline

Mental health multi-task classification model training pipeline based on XLM-RoBERTa Large.

## Model Architecture

The model uses XLM-RoBERTa Large as the backbone with 5 prediction heads:

| Head | Type | Range | Description |
|------|------|-------|-------------|
| Sentiment | Regression | -1 to 1 | Overall emotional tone |
| Family History | Binary | 0 or 1 | Indicates family mental health history |
| Trauma | Regression | 0 to 7 | Level of trauma indicators |
| Isolation | Regression | 0 to 4 | Social isolation score |
| Support | Regression | 0 to ~0.04 | Support system strength |

## Dataset

Training uses the `phoenix1803/Mental-Health-LongParas` dataset from HuggingFace:
- 120,000 rows of mental health narratives
- 80/20 train/test split (stratified by label)
- Labels: Anxiety, Depression, Stress, Bipolar, Personality disorder, etc.

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Dataset loading and PyTorch Dataset class |
| `model.py` | MultiTaskModel architecture and utilities |
| `train.py` | Main training script with CLI arguments |
| `inference.py` | FastAPI inference service |
| `requirements.txt` | Python dependencies |
| `Dockerfile.train` | Container for training jobs |
| `Dockerfile.inference` | Container for inference service |

## Local Development

### Setup

```bash
cd ml-training
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training (Local)

```bash
# Full training (requires GPU)
python train.py --epochs 2 --batch-size 32 --output-dir ./trained_model

# Quick test (CPU, 1 epoch, small batch)
python train.py --epochs 1 --batch-size 8 --output-dir ./trained_model
```

### Inference (Local)

```bash
# Start inference server
uvicorn inference:app --host 0.0.0.0 --port 8080

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel overwhelmed and alone.", "return_all_scores": true}'
```

## Docker Build

### Training Image

```bash
# Build training image
docker buildx build \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/project-4fc52960-1177-49ec-a6f/bloom-images/ml-training:latest \
  -f Dockerfile.train \
  --push .
```

### Inference Image

```bash
# After training, copy model to trained_model/ directory
docker buildx build \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/project-4fc52960-1177-49ec-a6f/bloom-images/ml-inference:latest \
  -f Dockerfile.inference \
  --push .
```

## GKE Deployment

### Running Training Job

```bash
# Apply training job (requires GPU node pool)
kubectl apply -f ../k8s/base/ml-training-job.yaml -n bloom-qa

# For CPU-only training (slower but works on Autopilot)
kubectl apply -f ../k8s/base/ml-training-job.yaml -n bloom-qa
# Then: kubectl delete job ml-training -n bloom-qa
# And create job from the ml-training-cpu spec

# Monitor training
kubectl logs -f job/ml-training -n bloom-qa
```

### Deploy Inference Service

```bash
# Deploy inference service
kubectl apply -f ../k8s/base/ml-deployment.yaml -n bloom-qa

# Check status
kubectl get pods -n bloom-qa -l app=ml-inference
kubectl logs -f deployment/ml-inference -n bloom-qa
```

## GCS Model Storage

Models are stored in `gs://bloom-health-ml-models/`:

```
bloom-health-ml-models/
├── latest/                    # Symlink to current production model
│   ├── model_vXXXXXX.pt
│   └── tokenizer files...
├── v20241207_120000/          # Timestamped versions
│   ├── model_v20241207_120000.pt
│   └── tokenizer files...
└── v20241206_090000/
    └── ...
```

### Upload Model Manually

```bash
# After training
gsutil -m cp -r trained_model/* gs://bloom-health-ml-models/v20241207_120000/

# Update latest symlink
gsutil -m rsync -r gs://bloom-health-ml-models/v20241207_120000/ gs://bloom-health-ml-models/latest/
```

## API Endpoints

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "labels": ["Anxiety", "Depression", ...]
}
```

### Single Prediction
```
POST /predict
Body: {
  "text": "I feel overwhelmed and alone.",
  "return_all_scores": false
}
Response: {
  "prediction": {
    "label": "Depression",
    "confidence": 0.85,
    "risk_level": "medium",
    "psychometrics": {
      "sentiment": -0.45,
      "trauma": 2.3,
      "isolation": 2.8,
      "support": 0.01,
      "family_history_prob": 0.12
    }
  }
}
```

### Batch Prediction
```
POST /predict/batch
Body: {
  "texts": ["text1", "text2", ...],
  "return_all_scores": false
}
Response: {
  "predictions": [...]
}
```

## Performance

Training on NVIDIA A100 (80GB):
- 120k rows, 2 epochs
- Batch size: 32
- Time: ~30 minutes
- Best validation loss: 0.2417

Inference:
- Single prediction: ~50ms (GPU), ~500ms (CPU)
- Batch of 32: ~200ms (GPU), ~2s (CPU)

## Switching from Mock to Real ML

In the Bloom app, set these environment variables:

```bash
# For mock mode (development)
USE_MOCK_ML=true

# For real ML (production)
USE_MOCK_ML=false
ML_INFERENCE_URL=http://ml-inference:8080
```
