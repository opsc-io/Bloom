"""
Data Preprocessing Script for Mental Health Analysis Model
Cleans, tokenizes, and prepares data for training.

CRISP-DM Phase: Data Preparation
"""

import os
import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from google.cloud import storage


# Preprocessing constants
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 512
TOKENIZER_NAME = "xlm-roberta-large"

# Label columns from the dataset
LABEL_COLUMNS = [
    "Sentiment",      # -1 to 1
    "Trauma",         # 0 to 7
    "Isolation",      # 0 to 4
    "Support",        # 0 to 1
    "Family_History", # 0 or 1 (binary)
]

GCS_BUCKET = "bloom-ml-data"


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove [deleted] and [removed] markers
    text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove excessive punctuation
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)

    return text


def filter_data(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Filter out low-quality samples."""
    original_len = len(df)

    # Remove empty or too short texts
    df = df[df[text_column].str.len() >= MIN_TEXT_LENGTH]

    # Remove texts that are just [deleted]
    df = df[~df[text_column].str.lower().str.contains(r'^\s*\[deleted\]\s*$', regex=True, na=True)]

    print(f"Filtered: {original_len} -> {len(df)} rows ({original_len - len(df)} removed)")
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label columns to expected ranges."""
    if "Sentiment" in df.columns:
        # Ensure sentiment is in [-1, 1]
        df["Sentiment"] = df["Sentiment"].clip(-1, 1)

    if "Trauma" in df.columns:
        # Ensure trauma is in [0, 7]
        df["Trauma"] = df["Trauma"].clip(0, 7)

    if "Isolation" in df.columns:
        # Ensure isolation is in [0, 4]
        df["Isolation"] = df["Isolation"].clip(0, 4)

    if "Support" in df.columns:
        # Ensure support is in [0, 1]
        df["Support"] = df["Support"].clip(0, 1)

    if "Family_History" in df.columns:
        # Binary classification
        df["Family_History"] = (df["Family_History"] > 0.5).astype(int)

    return df


def tokenize_dataset(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Add tokenized fields for model training."""
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Tokenizing texts...")
    # Tokenize with truncation
    encodings = tokenizer(
        df[text_column].tolist(),
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        padding=False,
        return_attention_mask=True,
    )

    df["input_ids"] = encodings["input_ids"]
    df["attention_mask"] = encodings["attention_mask"]
    df["token_count"] = [len(ids) for ids in encodings["input_ids"]]

    print(f"Tokenization complete. Avg tokens: {df['token_count'].mean():.1f}")
    return df


def create_splits(df: pd.DataFrame,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> tuple:
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        random_state=seed
    )

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = GCS_BUCKET):
    """Upload file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess mental health data")
    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file path")
    parser.add_argument("--output-dir", type=str, default="./data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Name of text column in input data")
    parser.add_argument("--upload-gcs", action="store_true",
                        help="Upload processed data to GCS")
    parser.add_argument("--version", type=str, default="v1",
                        help="Dataset version for GCS path")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Skip tokenization step")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    # Clean text
    print("\nCleaning text...")
    df[args.text_column] = df[args.text_column].apply(clean_text)

    # Filter data
    print("\nFiltering data...")
    df = filter_data(df, args.text_column)

    # Normalize labels
    print("\nNormalizing labels...")
    df = normalize_labels(df)

    # Tokenize (optional)
    if not args.skip_tokenize:
        print("\nTokenizing...")
        df = tokenize_dataset(df, args.text_column)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = create_splits(df)

    # Save splits
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        local_path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(local_path, index=False)
        print(f"Saved {split_name} to {local_path}")

        if args.upload_gcs:
            gcs_path = f"processed/{args.version}/{split_name}.parquet"
            upload_to_gcs(str(local_path), gcs_path)

    # Save preprocessing metadata
    metadata = {
        "version": args.version,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "tokenizer": TOKENIZER_NAME if not args.skip_tokenize else None,
        "max_length": MAX_TEXT_LENGTH,
        "label_columns": LABEL_COLUMNS,
    }

    metadata_path = output_dir / "metadata.json"
    pd.Series(metadata).to_json(metadata_path)
    print(f"\nMetadata saved to {metadata_path}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
