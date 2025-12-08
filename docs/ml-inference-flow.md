# ML Inference Flow Documentation

## Overview

The Bloom Health app uses a multi-task ML model to analyze patient messages for mental health indicators. The model runs on Google Cloud Vertex AI and provides real-time psychometric analysis to therapists.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ML INFERENCE FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐
│   Patient    │     │   Next.js    │     │         Vertex AI                │
│   Message    │────▶│   App (GKE)  │────▶│    Endpoint (us-central1)        │
│              │     │              │     │                                  │
│ "i feel      │     │ ml-inference │     │  XLM-RoBERTa Model               │
│  terrible"   │     │     .ts      │     │  (Frozen backbone + 5 heads)     │
└──────────────┘     └──────┬───────┘     └───────────────┬──────────────────┘
                           │                              │
                           │                              ▼
                           │              ┌───────────────────────────────────┐
                           │              │     MODEL OUTPUTS (Raw)           │
                           │              │  ┌─────────────────────────────┐  │
                           │              │  │ sentiment:  -0.35           │  │
                           │              │  │ trauma:      0.52           │  │
                           │              │  │ isolation:   0.41           │  │
                           │              │  │ support:     0.28           │  │
                           │              │  │ family_prob: 0.15           │  │
                           │              │  └─────────────────────────────┘  │
                           │              └───────────────┬───────────────────┘
                           │                              │
                           ▼                              ▼
              ┌────────────────────────────────────────────────────────────────┐
              │              CLIENT-SIDE LABEL RECALCULATION                   │
              │                                                                │
              │   determineLabelFromPsychometrics(psychometrics)               │
              │                                                                │
              │   sentiment = -0.35  (< -0.2)  ──▶  Check trauma/isolation     │
              │   trauma = 0.52      (> 0.5)   ──▶  Anxiety threshold met      │
              │   isolation = 0.41   (> 0.4)                                   │
              │                                                                │
              │   RESULT: { label: "Anxiety", confidence: 0.66 }               │
              └────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
              ┌────────────────────────────────────────────────────────────────┐
              │                    FINAL RESPONSE                              │
              │  ┌──────────────────────────────────────────────────────────┐  │
              │  │  label: "Anxiety"                                        │  │
              │  │  confidence: 0.66                                        │  │
              │  │  riskLevel: "medium"                                     │  │
              │  │  psychometrics: { sentiment, trauma, isolation, ... }    │  │
              │  └──────────────────────────────────────────────────────────┘  │
              └────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
              ┌────────────────────────────────────────────────────────────────┐
              │                 STORED & BROADCAST                             │
              │                                                                │
              │  1. Save to Database (MessageAnalysis table)                   │
              │  2. Publish to Redis: analysis:{conversationId}                │
              │  3. Socket.io broadcasts to therapist                          │
              └────────────────────────────────────────────────────────────────┘
```

## Model Architecture

### Base Model
- **Model**: XLM-RoBERTa Large (multilingual transformer)
- **Hidden Size**: 1024
- **Training**: Backbone frozen, only prediction heads trained

### Prediction Heads (5 outputs)
| Head | Output Range | Description |
|------|--------------|-------------|
| `sentiment` | -1 to 1 | Negative to positive sentiment |
| `trauma` | 0 to 7 | Trauma indicator score (typically 0.3-0.8) |
| `isolation` | 0 to 4 | Social isolation score (typically 0.2-0.7) |
| `support` | 0 to 1 | Support system strength |
| `family_history` | 0 to 1 | Probability of family mental health history |

### Training Dataset
- **Source**: `phoenix1803/Mental-Health-LongParas` (HuggingFace)
- **Split**: 70% train, 15% validation, 15% test
- **Loss Functions**: MSE for regression heads, BCE for family_history

## Label Determination Logic

The model predicts psychometric scores, not labels directly. Labels are determined by the `determineLabelFromPsychometrics()` function:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    determineLabelFromPsychometrics()                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  sentiment < -0.5  ──────────────────────────────────────────────────────┐  │
│      │                                                                   │  │
│      ├── trauma > 0.6 OR isolation > 0.5  ──▶  "Suicidal" (high risk)    │  │
│      └── else                             ──▶  "Depression"              │  │
│                                                                          │  │
│  sentiment < -0.2  ──────────────────────────────────────────────────────┤  │
│      │                                                                   │  │
│      ├── trauma > 0.5 AND isolation > 0.4 ──▶  "Depression"              │  │
│      ├── trauma > 0.4 OR isolation > 0.4  ──▶  "Anxiety"                 │  │
│      └── else                             ──▶  "Stress"                  │  │
│                                                                          │  │
│  sentiment < 0  ─────────────────────────────────────────────────────────┤  │
│      │                                                                   │  │
│      ├── trauma > 0.5 OR isolation > 0.5  ──▶  "Anxiety"                 │  │
│      └── else                             ──▶  "Stress"                  │  │
│                                                                          │  │
│  sentiment > 0.3  ───────────────────────────────────────────────────────┤  │
│      │                                                                   │  │
│      ├── trauma < 0.4 AND isolation < 0.4 ──▶  "Normal"                  │  │
│      ├── trauma > 0.5                     ──▶  "Bipolar"                 │  │
│      └── else                             ──▶  "Normal"                  │  │
│                                                                          │  │
│  sentiment 0 to 0.3  ────────────────────────────────────────────────────┤  │
│      │                                                                   │  │
│      ├── trauma > 0.5 OR isolation > 0.5  ──▶  "Stress"                  │  │
│      └── else                             ──▶  "Normal"                  │  │
│                                                                          │  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Mental Health Labels

| Label | Risk Level | Description |
|-------|------------|-------------|
| `Normal` | normal | No significant mental health indicators |
| `Stress` | low | Mild stress indicators |
| `Anxiety` | medium | Anxiety symptoms detected |
| `Depression` | medium/high | Depression indicators (high if confidence > 0.7) |
| `Bipolar` | medium | Mood instability indicators |
| `Personality disorder` | medium | Personality disorder traits |
| `Suicidal` | **high** | Suicidal ideation detected |

## Risk Level Determination

```typescript
function getRiskLevelFromPrediction(label, confidence) {
  if (label === "Suicidal" && confidence > 0.5) return "high";
  if (label === "Depression" && confidence > 0.7) return "high";
  if (["Depression", "Anxiety", "Bipolar"].includes(label) && confidence > 0.5) return "medium";
  if (label === "Normal") return "normal";
  return "low";
}
```

## Expected Results

| Message | Sentiment | Trauma | Isolation | Label | Risk |
|---------|-----------|--------|-----------|-------|------|
| "i feel terrible" | -0.3 | 0.55 | 0.45 | Anxiety | medium |
| "i feel awesome" | 0.5 | 0.35 | 0.30 | Normal | normal |
| "not feeling well" | -0.15 | 0.48 | 0.42 | Stress | low |
| "im feeling too good" | 0.6 | 0.32 | 0.28 | Normal | normal |
| "i want to end it all" | -0.7 | 0.65 | 0.55 | Suicidal | high |

## Infrastructure

### Vertex AI Endpoint
- **Endpoint ID**: `7919358942893834240`
- **Model ID**: `1495941644682264576` (v2)
- **Region**: `us-central1`
- **Project**: `project-4fc52960-1177-49ec-a6f`
- **Machine Type**: `n1-standard-4`
- **Replicas**: 1-2 (autoscaling)

### GKE Deployment
- **Cluster**: `bloom-dev-autopilot` (us-west1)
- **Namespace**: `bloom-dev`
- **Service Account**: `bloom-sa` (Workload Identity enabled)
- **GCP SA**: `bloom-app-sa@project-4fc52960-1177-49ec-a6f.iam.gserviceaccount.com`

### Authentication Flow
1. GKE pod uses Workload Identity
2. K8s SA `bloom-sa` impersonates GCP SA `bloom-app-sa`
3. GCP SA has `roles/aiplatform.user` for Vertex AI access
4. Token fetched from GCE metadata server: `http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`

## Key Files

| File | Purpose |
|------|---------|
| `src/lib/ml-inference.ts` | Client-side ML inference with label recalculation |
| `ml-training/inference.py` | Vertex AI container inference service |
| `ml-training/model.py` | Multi-task model architecture |
| `ml-training/train.py` | Training script |
| `ml-training/dataset.py` | Dataset loading and preprocessing |
| `src/app/api/messages/route.ts` | API endpoint that triggers ML analysis |
| `scripts/socket-server.ts` | Real-time analysis broadcast to therapists |

## Data Flow

1. **Patient sends message** → POST `/api/messages`
2. **Message saved** → Database (Message table)
3. **ML analysis triggered** → `analyzeText()` in `ml-inference.ts`
4. **Vertex AI called** → Returns psychometric scores
5. **Label recalculated** → `determineLabelFromPsychometrics()`
6. **Analysis saved** → Database (MessageAnalysis table)
7. **Redis publish** → `analysis:{conversationId}` channel
8. **Socket broadcast** → Only to therapists in the conversation
9. **Frontend updates** → Therapist sees analysis in real-time

---

## Model Training Pipeline

### Overview

The ML model is trained using Vertex AI Custom Training Jobs. The pipeline handles data loading, model training, and artifact storage to Google Cloud Storage.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ML TRAINING PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────────┐
│  HuggingFace     │     │   Vertex AI      │     │   Google Cloud Storage       │
│  Dataset         │────▶│   Training Job   │────▶│   (Model Artifacts)          │
│                  │     │                  │     │                              │
│ phoenix1803/     │     │  GPU: T4/A100    │     │  gs://bloom-ml-training/     │
│ Mental-Health-   │     │  Region: us-c1   │     │    └── trained_model/       │
│ LongParas        │     │                  │     │        ├── model_v1.pt      │
└──────────────────┘     └────────┬─────────┘     │        ├── tokenizer/       │
                                  │               │        └── config.json      │
                                  │               └──────────────┬───────────────┘
                                  │                              │
                                  ▼                              ▼
                    ┌─────────────────────────┐    ┌─────────────────────────────┐
                    │   Training Process      │    │   Model Upload to           │
                    │                         │    │   Vertex AI Model Registry  │
                    │  1. Load XLM-RoBERTa    │    │                             │
                    │  2. Freeze backbone     │    │  Model ID: 1495941644...    │
                    │  3. Train 5 heads       │    │  Version: v2                │
                    │  4. Validate & Save     │    │                             │
                    └─────────────────────────┘    └──────────────┬──────────────┘
                                                                  │
                                                                  ▼
                                                  ┌─────────────────────────────┐
                                                  │   Deploy to Endpoint        │
                                                  │                             │
                                                  │  Endpoint: 7919358942...    │
                                                  │  Machine: n1-standard-4     │
                                                  │  Replicas: 1-2 (autoscale)  │
                                                  └─────────────────────────────┘
```

### Training Steps

#### 1. Data Preparation (`ml-training/dataset.py`)

```python
# Load from HuggingFace
dataset = load_dataset("phoenix1803/Mental-Health-LongParas")

# Split: 70% train, 15% val, 15% test
train_df, val_df, test_df = train_test_split(...)

# Target columns:
# - sentiment_intensity: -1 to 1
# - family_history: 0 or 1
# - trauma_indicators: 0 to 7
# - social_isolation_score: 0 to 4
# - support_system_strength: 0 to ~0.04 (scaled by 100 for training)
```

#### 2. Model Architecture (`ml-training/model.py`)

```python
class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model, dropout_rate=0.1, freeze_backbone=True):
        # XLM-RoBERTa Large backbone (frozen)
        self.backbone = pretrained_model

        # 5 prediction heads (trainable)
        self.head_sentiment = nn.Linear(1024, 1)   # Regression
        self.head_family = nn.Linear(1024, 1)      # Binary (logit)
        self.head_trauma = nn.Linear(1024, 1)      # Regression
        self.head_isolation = nn.Linear(1024, 1)   # Regression
        self.head_support = nn.Linear(1024, 1)     # Regression
```

#### 3. Training Script (`ml-training/train.py`)

```python
# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 256

# Loss functions
loss_sentiment = MSELoss()      # Regression
loss_family = BCEWithLogitsLoss()  # Binary classification
loss_trauma = MSELoss()         # Regression
loss_isolation = MSELoss()      # Regression
loss_support = MSELoss()        # Regression

# Combined loss (weighted)
total_loss = (
    loss_sentiment * 1.0 +
    loss_family * 0.5 +
    loss_trauma * 1.0 +
    loss_isolation * 1.0 +
    loss_support * 0.5
)
```

#### 4. Inference Service (`ml-training/inference.py`)

```python
# FastAPI endpoints
POST /predict          # Single text prediction
POST /predict/batch    # Batch prediction
GET  /health          # Health check

# Vertex AI format support
{
  "instances": [{"text": "...", "return_all_scores": true}]
}
# Returns:
{
  "predictions": [{
    "label": "Anxiety",
    "confidence": 0.75,
    "risk_level": "medium",
    "psychometrics": {...}
  }]
}
```

### Running Training Locally

```bash
cd ml-training

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Test inference locally
python inference.py
# Then: curl -X POST http://localhost:8080/predict \
#   -H "Content-Type: application/json" \
#   -d '{"text": "I feel terrible"}'
```

### Running Training on Vertex AI

```bash
# Build and push Docker image
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-training:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-training:latest

# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=bloom-ml-training \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-training:latest

# Monitor training
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Uploading Model to Vertex AI

```bash
# 1. Upload trained model to GCS
gsutil -m cp -r ./trained_model gs://bloom-ml-training/models/v2/

# 2. Register model in Vertex AI
gcloud ai models upload \
  --region=us-central1 \
  --display-name=bloom-mental-health-v2 \
  --container-image-uri=us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-inference:latest \
  --artifact-uri=gs://bloom-ml-training/models/v2/

# 3. Deploy to endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=bloom-model-v2 \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=2
```

### GCS Bucket Structure

```
gs://bloom-ml-training/
├── datasets/
│   └── mental-health-longparas/
├── models/
│   ├── v1/
│   │   ├── model_v1.pt
│   │   ├── tokenizer/
│   │   └── config.json
│   └── v2/
│       ├── model_v1.pt
│       ├── tokenizer/
│       └── config.json
├── logs/
│   └── training-runs/
└── checkpoints/
```

### Environment Variables for Training

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Base transformer model | `xlm-roberta-large` |
| `GCS_BUCKET` | GCS bucket for artifacts | `bloom-ml-training` |
| `EPOCHS` | Training epochs | `3` |
| `BATCH_SIZE` | Training batch size | `16` |
| `LEARNING_RATE` | Learning rate | `2e-4` |
| `MAX_LENGTH` | Max sequence length | `256` |
| `FREEZE_BACKBONE` | Freeze transformer backbone | `true` |

### Monitoring & Logs

```bash
# View training job status
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# View endpoint metrics
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Model Versioning

| Version | Model ID | Changes | Date |
|---------|----------|---------|------|
| v1 | `1651878781779968000` | Initial training | 2024-12-06 |
| v2 | `1495941644682264576` | Improved thresholds | 2024-12-08 |

### Costs

| Resource | Cost Estimate |
|----------|---------------|
| Training (T4 GPU, 2hr) | ~$2-5 per run |
| Endpoint (n1-standard-4) | ~$0.19/hour |
| GCS Storage | ~$0.02/GB/month |

---

## Troubleshooting

### Common Issues

1. **All predictions return "Stress"**
   - Cause: Label determination thresholds too high
   - Fix: Updated `determineLabelFromPsychometrics()` with realistic thresholds

2. **Vertex AI 403 Permission Denied**
   - Cause: Missing IAM permissions
   - Fix: Enable Workload Identity, grant `roles/aiplatform.user` to service account

3. **Model outputs same psychometrics for all inputs**
   - Cause: Frozen backbone not learning domain-specific features
   - Fix: Consider unfreezing backbone or training longer

4. **Slow inference**
   - Cause: Cold start on autoscaled endpoint
   - Fix: Set `min-replica-count=1` to keep warm instance
