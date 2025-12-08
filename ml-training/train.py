#!/usr/bin/env python3
"""
Mental Health Multi-Task Model Training Script
Vertex AI MLOps Compatible

Based on notebook: Final_2.ipynb
- Frozen backbone, train only heads
- 5 epochs with LR decay
- pos_weight for imbalanced family_history
- 70/30 train/val+test split

Usage (Local):
    python train.py --epochs 5 --batch-size 32 --output-dir ./trained_model

Usage (Vertex AI):
    The script automatically detects Vertex AI environment and uses:
    - AIP_MODEL_DIR for model output
    - AIP_TENSORBOARD_LOG_DIR for TensorBoard logs
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MentalHealthDataset, load_mental_health_data
from model import (
    create_multitask_model,
    load_model,
    save_model,
    DEFAULT_MODEL_NAME
)

# Vertex AI environment variables
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR")
AIP_TENSORBOARD_LOG_DIR = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
CLOUD_ML_PROJECT_ID = os.environ.get("CLOUD_ML_PROJECT_ID")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-task mental health classification model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (higher for heads-only training)"
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.5,
        help="Learning rate decay factor per epoch"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--freeze-backbone",
        type=bool,
        default=True,
        help="Freeze backbone and train only heads (faster)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_model",
        help="Directory to save trained model (overridden by AIP_MODEL_DIR on Vertex AI)"
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket to upload model (e.g., gs://bloom-health-ml-models)"
    )
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch: int, initial_lr: float, decay: float):
    """Decay learning rate after each epoch (halves LR after epoch 0)."""
    if epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay
        print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.2e}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mse_loss_fn: nn.Module,
    bce_loss_fn: nn.Module,
    use_amp: bool = True
) -> tuple:
    """Train for one epoch. Returns (total_loss, per_head_losses)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    head_loss_sum = {
        "sentiment": 0.0,
        "trauma": 0.0,
        "isolation": 0.0,
        "support": 0.0,
        "family": 0.0,
    }

    progress = tqdm(loader, desc="Training", leave=True)
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Prepare targets
        targets = {
            k: v.to(device).unsqueeze(1)
            for k, v in batch.items()
            if k not in ['input_ids', 'attention_mask']
        }

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision forward pass
            with autocast():
                outputs = model(input_ids, attention_mask)

                # Per-head losses
                loss_sent = mse_loss_fn(outputs['sentiment'], targets['sentiment'])
                loss_trauma = mse_loss_fn(outputs['trauma'], targets['trauma'])
                loss_iso = mse_loss_fn(outputs['isolation'], targets['isolation'])
                loss_sup = mse_loss_fn(outputs['support'], targets['support'])
                loss_fam = bce_loss_fn(outputs['family'], targets['family'])

                # Combined loss
                loss = loss_sent + loss_trauma + loss_iso + loss_sup + loss_fam

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)

            loss_sent = mse_loss_fn(outputs['sentiment'], targets['sentiment'])
            loss_trauma = mse_loss_fn(outputs['trauma'], targets['trauma'])
            loss_iso = mse_loss_fn(outputs['isolation'], targets['isolation'])
            loss_sup = mse_loss_fn(outputs['support'], targets['support'])
            loss_fam = bce_loss_fn(outputs['family'], targets['family'])

            loss = loss_sent + loss_trauma + loss_iso + loss_sup + loss_fam

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        head_loss_sum["sentiment"] += loss_sent.item()
        head_loss_sum["trauma"] += loss_trauma.item()
        head_loss_sum["isolation"] += loss_iso.item()
        head_loss_sum["support"] += loss_sup.item()
        head_loss_sum["family"] += loss_fam.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_total = total_loss / max(n_batches, 1)
    avg_heads = {k: v / max(n_batches, 1) for k, v in head_loss_sum.items()}
    return avg_total, avg_heads


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse_loss_fn: nn.Module,
    bce_loss_fn: nn.Module
) -> tuple:
    """Evaluate model on validation set. Returns (total_loss, per_head_losses)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    head_loss_sum = {
        "sentiment": 0.0,
        "trauma": 0.0,
        "isolation": 0.0,
        "support": 0.0,
        "family": 0.0,
    }

    for batch in tqdm(loader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        targets = {
            k: v.to(device).unsqueeze(1)
            for k, v in batch.items()
            if k not in ['input_ids', 'attention_mask']
        }

        outputs = model(input_ids, attention_mask)

        loss_sent = mse_loss_fn(outputs['sentiment'], targets['sentiment'])
        loss_trauma = mse_loss_fn(outputs['trauma'], targets['trauma'])
        loss_iso = mse_loss_fn(outputs['isolation'], targets['isolation'])
        loss_sup = mse_loss_fn(outputs['support'], targets['support'])
        loss_fam = bce_loss_fn(outputs['family'], targets['family'])

        loss = loss_sent + loss_trauma + loss_iso + loss_sup + loss_fam

        total_loss += loss.item()
        n_batches += 1

        head_loss_sum["sentiment"] += loss_sent.item()
        head_loss_sum["trauma"] += loss_trauma.item()
        head_loss_sum["isolation"] += loss_iso.item()
        head_loss_sum["support"] += loss_sup.item()
        head_loss_sum["family"] += loss_fam.item()

    avg_total = total_loss / max(n_batches, 1)
    avg_heads = {k: v / max(n_batches, 1) for k, v in head_loss_sum.items()}
    return avg_total, avg_heads


def upload_to_gcs(local_dir: str, gcs_path: str):
    """Upload model to Google Cloud Storage."""
    from google.cloud import storage

    print(f"Uploading model to {gcs_path}")

    # Parse GCS path
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload all files
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_name = f"{prefix}/{rel_path}" if prefix else rel_path

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"  Uploaded: gs://{bucket_name}/{blob_name}")

    print(f"Model uploaded to gs://{bucket_name}/{prefix}/")


def save_metrics(output_dir: str, metrics: dict):
    """Save training metrics for Vertex AI."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    args = parse_args()

    # Determine output directory (Vertex AI override)
    output_dir = AIP_MODEL_DIR if AIP_MODEL_DIR else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("Mental Health Multi-Task Model Training")
    print("=" * 60)
    print(f"Environment: {'Vertex AI' if AIP_MODEL_DIR else 'Local'}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate} (heads only)")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Max Length: {args.max_length}")
    print(f"Output Dir: {output_dir}")
    if args.gcs_bucket:
        print(f"GCS Bucket: {args.gcs_bucket}")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be slow!")
        device = torch.device("cpu")
        use_amp = False
    else:
        device = torch.device("cuda")
        use_amp = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data (70/30 with val/test split)
    print("\n--- Loading Data ---")
    train_df, val_df, test_df = load_mental_health_data()

    # Load model and tokenizer
    print("\n--- Loading Model ---")
    tokenizer, base_model, device = load_model(args.model_name, device)
    model = create_multitask_model(
        base_model,
        device,
        freeze_backbone=args.freeze_backbone
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Create datasets and loaders
    print("\n--- Creating DataLoaders ---")
    train_dataset = MentalHealthDataset(train_df, tokenizer, args.max_length)
    val_dataset = MentalHealthDataset(val_df, tokenizer, args.max_length)
    test_dataset = MentalHealthDataset(test_df, tokenizer, args.max_length)

    # Adjust num_workers based on environment
    num_workers = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Compute pos_weight for imbalanced family_history
    pos_count = (train_df['family_history'] == 1).sum()
    neg_count = (train_df['family_history'] == 0).sum()
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"family_history pos_weight: {pos_weight_val:.2f}")

    # Loss functions
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device)
    )

    # Optimizer - only train head parameters (backbone is frozen)
    head_params = [p for name, p in model.named_parameters()
                   if p.requires_grad and 'head_' in name]
    print(f"Trainable tensors (heads only): {len(head_params)}")
    optimizer = AdamW(head_params, lr=args.learning_rate)

    scaler = GradScaler(enabled=use_amp)

    # Training loop
    print("\n--- Starting Training ---")
    best_loss = float('inf')
    best_weights = None
    training_start = time.time()

    epoch_metrics = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_decay)

        # Train and evaluate
        train_loss, train_heads = train_epoch(
            model, train_loader, optimizer, scaler, device,
            mse_loss_fn, bce_loss_fn, use_amp
        )
        val_loss, val_heads = evaluate(model, val_loader, device, mse_loss_fn, bce_loss_fn)

        epoch_time = time.time() - epoch_start
        print(f"Results: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"Per-head losses (train):")
        for k, v in train_heads.items():
            print(f"  {k}: {v:.4f}")
        print(f"Epoch Time: {epoch_time / 60:.1f} minutes")

        # Track metrics
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_heads": train_heads,
            "val_heads": val_heads,
            "epoch_time_seconds": epoch_time
        })

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print("New best model saved!")

    # Restore best weights
    model.load_state_dict(best_weights)
    total_time = time.time() - training_start

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Total Training Time: {total_time / 60:.1f} minutes")
    print("=" * 60)

    # Final evaluation on test set
    print("\n--- Final Test Evaluation ---")
    test_loss, test_heads = evaluate(model, test_loader, device, mse_loss_fn, bce_loss_fn)
    print(f"Test Loss: {test_loss:.4f}")
    for k, v in test_heads.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    print(f"\n--- Saving Model (version: {model_version}) ---")
    save_model(model, tokenizer, output_dir, model_version)

    # Save metrics
    final_metrics = {
        "model_version": model_version,
        "model_name": args.model_name,
        "best_val_loss": best_loss,
        "test_loss": test_loss,
        "test_heads": test_heads,
        "total_training_time_seconds": total_time,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "freeze_backbone": args.freeze_backbone,
        "max_length": args.max_length,
        "pos_weight_family": pos_weight_val,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "epoch_metrics": epoch_metrics,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    save_metrics(output_dir, final_metrics)

    # Upload to GCS if specified (or if in Vertex AI)
    if args.gcs_bucket:
        gcs_path = f"{args.gcs_bucket}/{model_version}"
        upload_to_gcs(output_dir, gcs_path)

        # Also update "latest" symlink
        latest_path = f"{args.gcs_bucket}/latest"
        upload_to_gcs(output_dir, latest_path)

    print("\nDone!")
    return best_loss


if __name__ == "__main__":
    main()
