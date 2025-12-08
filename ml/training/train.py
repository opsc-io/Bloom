"""
Training Script for Mental Health Multi-Task Model
Runs on Vertex AI with GPU support.

CRISP-DM Phase: Modeling
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import r2_score, f1_score, mean_absolute_error
from google.cloud import storage
import hypertune


# Model configuration
MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 512
LABEL_COLUMNS = ["Sentiment", "Trauma", "Isolation", "Support", "Family_History"]


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for mental health text data."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.texts = df["text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Extract labels
        self.labels = {}
        for col in LABEL_COLUMNS:
            if col in df.columns:
                self.labels[col] = torch.tensor(df[col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

        for col, tensor in self.labels.items():
            item[col] = tensor[idx]

        return item


class MultiTaskModel(nn.Module):
    """Multi-task model with XLM-RoBERTa backbone and task-specific heads."""

    def __init__(self, pretrained_model_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Task-specific heads
        self.head_sentiment = nn.Linear(self.hidden_size, 1)   # -1 to 1
        self.head_trauma = nn.Linear(self.hidden_size, 1)      # 0 to 7
        self.head_isolation = nn.Linear(self.hidden_size, 1)   # 0 to 4
        self.head_support = nn.Linear(self.hidden_size, 1)     # 0 to 1
        self.head_family = nn.Linear(self.hidden_size, 1)      # binary (sigmoid)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)

        return {
            "sentiment": torch.tanh(self.head_sentiment(pooled)).squeeze(-1),  # [-1, 1]
            "trauma": torch.relu(self.head_trauma(pooled)).squeeze(-1) * 7,     # [0, 7]
            "isolation": torch.relu(self.head_isolation(pooled)).squeeze(-1) * 4, # [0, 4]
            "support": torch.sigmoid(self.head_support(pooled)).squeeze(-1),    # [0, 1]
            "family": torch.sigmoid(self.head_family(pooled)).squeeze(-1),      # [0, 1]
        }

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": MODEL_NAME,
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(pretrained_model_name=checkpoint.get("model_name", MODEL_NAME))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def compute_loss(predictions: dict, batch: dict) -> torch.Tensor:
    """Compute combined multi-task loss."""
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    loss = 0.0
    task_losses = {}

    # Regression tasks
    if "Sentiment" in batch:
        task_losses["sentiment"] = mse(predictions["sentiment"], batch["Sentiment"])
        loss += task_losses["sentiment"]

    if "Trauma" in batch:
        task_losses["trauma"] = mse(predictions["trauma"], batch["Trauma"])
        loss += task_losses["trauma"] * 0.5  # Scale since range is larger

    if "Isolation" in batch:
        task_losses["isolation"] = mse(predictions["isolation"], batch["Isolation"])
        loss += task_losses["isolation"]

    if "Support" in batch:
        task_losses["support"] = mse(predictions["support"], batch["Support"])
        loss += task_losses["support"]

    # Binary classification
    if "Family_History" in batch:
        task_losses["family"] = bce(predictions["family"], batch["Family_History"])
        loss += task_losses["family"]

    return loss, task_losses


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = {k: [] for k in ["sentiment", "trauma", "isolation", "support", "family"]}
    all_labels = {k: [] for k in LABEL_COLUMNS}
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            predictions = model(input_ids, attention_mask)

            # Move predictions to CPU
            for key, pred in predictions.items():
                all_preds[key].extend(pred.cpu().numpy())

            # Move labels to device and compute loss
            batch_device = {k: v.to(device) for k, v in batch.items() if k in LABEL_COLUMNS}
            loss, _ = compute_loss(predictions, batch_device)
            total_loss += loss.item()

            for col in LABEL_COLUMNS:
                if col in batch:
                    all_labels[col].extend(batch[col].numpy())

    # Compute metrics
    metrics = {"val_loss": total_loss / len(dataloader)}

    # R² for regression tasks
    label_to_pred = {
        "Sentiment": "sentiment",
        "Trauma": "trauma",
        "Isolation": "isolation",
        "Support": "support",
    }

    for label_col, pred_key in label_to_pred.items():
        if all_labels[label_col]:
            metrics[f"{pred_key}_r2"] = r2_score(all_labels[label_col], all_preds[pred_key])
            metrics[f"{pred_key}_mae"] = mean_absolute_error(all_labels[label_col], all_preds[pred_key])

    # F1 for family history (binary)
    if all_labels["Family_History"]:
        family_preds = [1 if p > 0.5 else 0 for p in all_preds["family"]]
        metrics["family_f1"] = f1_score(all_labels["Family_History"], family_preds)

    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        predictions = model(input_ids, attention_mask)

        # Move labels to device
        batch_device = {k: v.to(device) for k, v in batch.items() if k in LABEL_COLUMNS}
        loss, _ = compute_loss(predictions, batch_device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def download_from_gcs(gcs_path: str, local_path: str):
    """Download file from GCS."""
    client = storage.Client()

    # Parse gs:// path
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")


def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload file to GCS."""
    client = storage.Client()

    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")


def main():
    parser = argparse.ArgumentParser(description="Train mental health model")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to training data (local or gs://)")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Path to validation data (local or gs://)")
    parser.add_argument("--output-dir", type=str, default="/tmp/model",
                        help="Output directory for model")
    parser.add_argument("--model-gcs", type=str, default=None,
                        help="GCS path to upload trained model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download data if from GCS
    train_path = args.train_data
    val_path = args.val_data

    if args.train_data.startswith("gs://"):
        train_path = "/tmp/train.parquet"
        download_from_gcs(args.train_data, train_path)

    if args.val_data.startswith("gs://"):
        val_path = "/tmp/val.parquet"
        download_from_gcs(args.val_data, val_path)

    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Create tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MentalHealthDataset(train_df, tokenizer)
    val_dataset = MentalHealthDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    print("Creating model...")
    model = MultiTaskModel().to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_val_loss = float("inf")
    training_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Sentiment R²: {metrics.get('sentiment_r2', 'N/A'):.4f}")
        print(f"Family F1: {metrics.get('family_f1', 'N/A'):.4f}")

        # Report to Vertex AI hyperparameter tuning
        try:
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="val_loss",
                metric_value=metrics["val_loss"],
                global_step=epoch
            )
        except Exception:
            pass  # Not running in hyperparameter tuning mode

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics
        })

        # Save best model
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            model_path = output_dir / "best_model.pt"
            model.save(str(model_path))
            print(f"Saved best model to {model_path}")

    # Save final model and history
    final_path = output_dir / "final_model.pt"
    model.save(str(final_path))

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # Upload to GCS if specified
    if args.model_gcs:
        upload_to_gcs(str(output_dir / "best_model.pt"), f"{args.model_gcs}/best_model.pt")
        upload_to_gcs(str(history_path), f"{args.model_gcs}/training_history.json")

    print("\nTraining complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
