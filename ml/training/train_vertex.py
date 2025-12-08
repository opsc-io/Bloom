"""
Vertex AI Training Script for Mental Health Classification
Dataset: kamruzzaman-asif/reddit-mental-health-classification

This script runs on Vertex AI with GPU support.
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
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from google.cloud import storage

# Try to import hypertune for Vertex AI HPT
try:
    import hypertune
    HAS_HYPERTUNE = True
except ImportError:
    HAS_HYPERTUNE = False

# Model configuration
MODEL_NAME = "xlm-roberta-base"  # Using base for faster training
MAX_LENGTH = 256

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


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for Reddit mental health classification."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


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

    def save(self, path):
        """Save model weights and config."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": MODEL_NAME,
            "num_labels": self.num_labels,
            "labels": LABELS,
            "hidden_size": self.hidden_size,
        }, path)

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


def load_reddit_dataset():
    """Load the Reddit mental health classification dataset from Hugging Face."""
    print("Loading dataset: kamruzzaman-asif/reddit-mental-health-classification")

    dataset = load_dataset("kamruzzaman-asif/reddit-mental-health-classification")

    # Create label mapping
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}

    def process_split(split_data):
        texts = []
        labels = []
        for item in split_data:
            text = item.get("text", item.get("Text", ""))
            label = item.get("label", item.get("Label", "Normal"))

            if text and label in label_to_id:
                texts.append(text)
                labels.append(label_to_id[label])
        return texts, labels

    # Process train and test splits
    test_texts, test_labels = None, None

    if "train" in dataset:
        train_texts, train_labels = process_split(dataset["train"])
        # Check for test/validation split
        if "test" in dataset:
            test_texts, test_labels = process_split(dataset["test"])
        elif "validation" in dataset:
            test_texts, test_labels = process_split(dataset["validation"])
        else:
            # No test split - create from train data (20% for test)
            split_idx = int(0.8 * len(train_texts))
            test_texts, test_labels = train_texts[split_idx:], train_labels[split_idx:]
            train_texts, train_labels = train_texts[:split_idx], train_labels[:split_idx]
    else:
        # If no train split, use the full dataset and create splits
        full_texts, full_labels = process_split(dataset[list(dataset.keys())[0]])
        split_idx = int(0.8 * len(full_texts))
        train_texts, train_labels = full_texts[:split_idx], full_labels[:split_idx]
        test_texts, test_labels = full_texts[split_idx:], full_labels[split_idx:]

    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
    print(f"Label distribution (train): {np.bincount(train_labels, minlength=NUM_LABELS)}")

    return train_texts, train_labels, test_texts, test_labels


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy


def evaluate(model, dataloader, device, criterion):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "val_loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
    }

    return metrics, all_preds, all_labels


def upload_to_gcs(local_path, gcs_path):
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
    parser = argparse.ArgumentParser(description="Train mental health classifier on Vertex AI")
    parser.add_argument("--output-dir", type=str, default="/tmp/model",
                        help="Output directory for model")
    parser.add_argument("--model-gcs", type=str, default=None,
                        help="GCS path to upload trained model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_texts, train_labels, test_texts, test_labels = load_reddit_dataset()

    # Create validation split from training data
    val_size = int(0.1 * len(train_texts))
    val_texts, val_labels = train_texts[-val_size:], train_labels[-val_size:]
    train_texts, train_labels = train_texts[:-val_size], train_labels[:-val_size]

    print(f"Final splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Create tokenizer and datasets
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)
    test_dataset = MentalHealthDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    print("Creating model...")
    model = MentalHealthClassifier(dropout=args.dropout).to(device)

    # Loss function with class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=NUM_LABELS)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * NUM_LABELS
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_val_loss = float("inf")
    best_f1 = 0.0
    training_history = []

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_metrics, val_preds, val_labels_out = evaluate(model, val_loader, device, criterion)
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")

        # Report to Vertex AI hyperparameter tuning
        if HAS_HYPERTUNE:
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="val_loss",
                metric_value=val_metrics["val_loss"],
                global_step=epoch
            )
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="f1_macro",
                metric_value=val_metrics["f1_macro"],
                global_step=epoch
            )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            **val_metrics
        })

        # Save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_val_loss = val_metrics["val_loss"]
            model_path = output_dir / "best_model.pt"
            model.save(str(model_path))
            print(f"*** Saved best model (F1: {best_f1:.4f}) to {model_path}")

    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final Evaluation on Test Set")
    print(f"{'='*50}")

    # Load best model
    model = MentalHealthClassifier.load(str(output_dir / "best_model.pt"), device=device)
    model.to(device)

    test_metrics, test_preds, test_labels_out = evaluate(model, test_loader, device, criterion)
    print(f"Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels_out, test_preds, target_names=LABELS))

    # Save final model and history
    final_path = output_dir / "final_model.pt"
    model.save(str(final_path))

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # Save test results
    results = {
        "best_val_loss": best_val_loss,
        "best_f1": best_f1,
        "test_metrics": test_metrics,
        "labels": LABELS,
        "model_name": MODEL_NAME,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Upload to GCS if specified
    if args.model_gcs:
        print(f"\nUploading to GCS: {args.model_gcs}")
        upload_to_gcs(str(output_dir / "best_model.pt"), f"{args.model_gcs}/best_model.pt")
        upload_to_gcs(str(history_path), f"{args.model_gcs}/training_history.json")
        upload_to_gcs(str(results_path), f"{args.model_gcs}/results.json")

    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best F1 (macro): {best_f1:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
