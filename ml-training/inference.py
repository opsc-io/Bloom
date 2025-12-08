#!/usr/bin/env python3
"""
Mental Health ML Inference Service

FastAPI service for running inference on the trained multi-task model.
Provides real-time mental health text classification for the Bloom Health app.

Usage:
    uvicorn inference:app --host 0.0.0.0 --port 8080

Environment Variables:
    MODEL_DIR: Path to trained model directory (default: ./trained_model)
    MODEL_NAME: Base model name (default: xlm-roberta-large)
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from model import load_trained_model, DEFAULT_MODEL_NAME


# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "./trained_model")
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

# Global model state
model = None
tokenizer = None
device = None


# Mental health labels for primary classification
MENTAL_HEALTH_LABELS = [
    "Anxiety",
    "Depression",
    "Suicidal",
    "Stress",
    "Bipolar",
    "Personality disorder",
    "Normal",
]


# Request/Response models
class PredictRequest(BaseModel):
    text: str
    return_all_scores: bool = False


class BatchPredictRequest(BaseModel):
    texts: List[str]
    return_all_scores: bool = False


class PredictionResult(BaseModel):
    label: str
    confidence: float
    risk_level: str
    all_scores: Optional[Dict[str, float]] = None
    psychometrics: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    labels: List[str]


# FastAPI app
app = FastAPI(
    title="Bloom Health ML Inference",
    description="Mental health text classification API",
    version="1.0.0"
)


def determine_label(psychometrics: Dict[str, float]) -> tuple:
    """
    Determine primary mental health label from psychometric scores.

    Returns:
        Tuple of (label, confidence)
    """
    sentiment = psychometrics['sentiment']
    trauma = psychometrics['trauma']
    isolation = psychometrics['isolation']
    family_prob = psychometrics['family_history_prob']

    # High trauma + high isolation suggests Depression or Suicidal
    if trauma > 4 and isolation > 2:
        if sentiment < -0.5:
            return "Suicidal", min(0.95, 0.6 + trauma * 0.05)
        return "Depression", min(0.95, 0.5 + trauma * 0.05)

    # Moderate trauma/isolation suggests Anxiety or Stress
    if trauma > 2 or isolation > 1.5:
        if sentiment < -0.3:
            return "Anxiety", min(0.90, 0.5 + trauma * 0.05)
        return "Stress", min(0.85, 0.5 + isolation * 0.1)

    # Low indicators suggest Normal or mild conditions
    if sentiment > 0 and trauma < 1 and isolation < 1:
        return "Normal", min(0.90, 0.7 + sentiment * 0.2)

    # Default to Stress for ambiguous cases
    return "Stress", 0.5


def get_risk_level(label: str, confidence: float) -> str:
    """Determine risk level from label and confidence."""
    if label == "Suicidal" and confidence > 0.5:
        return "high"
    if label == "Depression" and confidence > 0.7:
        return "high"
    if label in ["Depression", "Anxiety", "Bipolar"] and confidence > 0.5:
        return "medium"
    if label == "Normal":
        return "normal"
    return "low"


class PsychometricLabeler:
    """
    Inference engine for the multi-task model.
    Generates psychometric profiles from text.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Predict psychometric profile for a single text.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])

        # Extract and process outputs
        sentiment = outputs['sentiment'].cpu().item()
        trauma = max(0, outputs['trauma'].cpu().item())
        isolation = max(0, outputs['isolation'].cpu().item())
        support = max(0, outputs['support'].cpu().item() / 100.0)

        # Sigmoid for family probability
        family_logit = outputs['family'].cpu().item()
        family_prob = 1 / (1 + np.exp(-family_logit))

        psychometrics = {
            'sentiment': float(sentiment),
            'trauma': float(trauma),
            'isolation': float(isolation),
            'support': float(support),
            'family_history_prob': float(family_prob)
        }

        # Determine primary label
        label, confidence = determine_label(psychometrics)
        risk_level = get_risk_level(label, confidence)

        return {
            'label': label,
            'confidence': confidence,
            'risk_level': risk_level,
            'psychometrics': psychometrics
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict psychometric profiles for multiple texts.
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(self.device)

            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])

            # Process each item in batch
            for j in range(len(batch_texts)):
                sentiment = outputs['sentiment'][j].cpu().item()
                trauma = max(0, outputs['trauma'][j].cpu().item())
                isolation = max(0, outputs['isolation'][j].cpu().item())
                support = max(0, outputs['support'][j].cpu().item() / 100.0)
                family_logit = outputs['family'][j].cpu().item()
                family_prob = 1 / (1 + np.exp(-family_logit))

                psychometrics = {
                    'sentiment': float(sentiment),
                    'trauma': float(trauma),
                    'isolation': float(isolation),
                    'support': float(support),
                    'family_history_prob': float(family_prob)
                }

                label, confidence = determine_label(psychometrics)
                risk_level = get_risk_level(label, confidence)

                all_results.append({
                    'label': label,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'psychometrics': psychometrics
                })

        return all_results


# Global labeler instance
labeler = None


@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the service starts."""
    global model, tokenizer, device, labeler

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Base model: {MODEL_NAME}")

    try:
        model, tokenizer, device = load_trained_model(
            MODEL_DIR,
            MODEL_NAME
        )
        labeler = PsychometricLabeler(model, tokenizer, device)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Service will return errors until model is available")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "none",
        labels=MENTAL_HEALTH_LABELS
    )


@app.post("/predict")
async def predict(request: Request):
    """
    Analyze text for mental health indicators.
    Supports both direct requests and Vertex AI format.

    Direct format: {"text": "...", "return_all_scores": true}
    Vertex AI format: {"instances": [{"text": "...", "return_all_scores": true}]}
    """
    if labeler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    body = await request.json()

    # Check if this is Vertex AI format (has "instances" key)
    if "instances" in body:
        # Vertex AI batch format - process all instances
        instances = body["instances"]
        predictions = []

        for instance in instances:
            text = instance.get("text", "")
            return_all_scores = instance.get("return_all_scores", False)

            result = labeler.predict(text)

            pred = {
                "label": result['label'],
                "confidence": result['confidence'],
                "risk_level": result['risk_level'],
            }

            if return_all_scores:
                pred["psychometrics"] = result['psychometrics']
                all_scores = {label: 0.1 for label in MENTAL_HEALTH_LABELS}
                all_scores[result['label']] = result['confidence']
                pred["all_scores"] = all_scores

            predictions.append(pred)

        # Vertex AI expects {"predictions": [...]}
        return {"predictions": predictions}

    # Direct format
    text = body.get("text", "")
    return_all_scores = body.get("return_all_scores", False)

    result = labeler.predict(text)

    # Build response
    prediction = PredictionResult(
        label=result['label'],
        confidence=result['confidence'],
        risk_level=result['risk_level'],
        psychometrics=result['psychometrics'] if return_all_scores else None,
        all_scores={
            label: 0.1 for label in MENTAL_HEALTH_LABELS
        } if return_all_scores else None
    )

    # Set the primary label score to confidence
    if prediction.all_scores:
        prediction.all_scores[result['label']] = result['confidence']

    return PredictResponse(prediction=prediction)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Analyze multiple texts in a batch.
    """
    if labeler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = labeler.predict_batch(request.texts)

    predictions = []
    for result in results:
        prediction = PredictionResult(
            label=result['label'],
            confidence=result['confidence'],
            risk_level=result['risk_level'],
            psychometrics=result['psychometrics'] if request.return_all_scores else None,
            all_scores=None
        )
        predictions.append(prediction)

    return BatchPredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
