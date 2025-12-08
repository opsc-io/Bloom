"""
FastAPI Inference Server for Mental Health Classification
Loads model from GCS and serves predictions via REST API.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
MODEL_GCS_PATH = os.environ.get("MODEL_GCS_PATH", "gs://bloom-health-ml-models/v1/best_model.pt")
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "/tmp/model.pt")
PORT = int(os.environ.get("PORT", 8080))

# Mental health classification labels
LABELS = [
    "Anxiety",
    "Depression",
    "Suicidal",
    "Stress",
    "Bipolar",
    "Personality disorder",
    "Normal"
]
NUM_LABELS = len(LABELS)

# Global model and tokenizer
model = None
tokenizer = None
device = None


class MentalHealthClassifier(nn.Module):
    """Mental health text classifier using transformer backbone."""

    def __init__(self, pretrained_model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    @classmethod
    def load(cls, path, device="cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            pretrained_model_name=checkpoint.get("model_name", MODEL_NAME),
            num_labels=checkpoint.get("num_labels", NUM_LABELS)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def download_model_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download model from GCS to local path."""
    logger.info(f"Downloading model from {gcs_path} to {local_path}")

    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

    logger.info(f"Model downloaded successfully")


def load_model():
    """Load model and tokenizer."""
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Download model from GCS if not exists locally
    if not Path(MODEL_LOCAL_PATH).exists():
        download_model_from_gcs(MODEL_GCS_PATH, MODEL_LOCAL_PATH)

    # Load model
    logger.info("Loading model...")
    model = MentalHealthClassifier.load(MODEL_LOCAL_PATH, device=device)
    model.to(device)
    model.eval()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    load_model()
    yield
    logger.info("Shutting down...")


# FastAPI app
app = FastAPI(
    title="Mental Health Classification API",
    description="Classify text into mental health categories",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class PredictRequest(BaseModel):
    text: str
    return_all_scores: bool = False


class PredictBatchRequest(BaseModel):
    texts: List[str]
    return_all_scores: bool = False


class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: Optional[dict] = None


class PredictResponse(BaseModel):
    prediction: PredictionResult


class PredictBatchResponse(BaseModel):
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    labels: List[str]


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        labels=LABELS
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict mental health category for a single text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize
        encoding = tokenizer(
            request.text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        result = PredictionResult(
            label=LABELS[pred_idx],
            confidence=confidence
        )

        if request.return_all_scores:
            result.all_scores = {
                LABELS[i]: probs[0][i].item()
                for i in range(len(LABELS))
            }

        return PredictResponse(prediction=result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """Predict mental health categories for multiple texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize all texts
        encoding = tokenizer(
            request.texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_indices = torch.argmax(probs, dim=1).cpu().numpy()

        predictions = []
        for i, pred_idx in enumerate(pred_indices):
            result = PredictionResult(
                label=LABELS[pred_idx],
                confidence=probs[i][pred_idx].item()
            )

            if request.return_all_scores:
                result.all_scores = {
                    LABELS[j]: probs[i][j].item()
                    for j in range(len(LABELS))
                }

            predictions.append(result)

        return PredictBatchResponse(predictions=predictions)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
