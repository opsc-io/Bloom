"""
Data Collection Script for Mental Health Analysis Model
Downloads and combines datasets from Hugging Face for training.

CRISP-DM Phase: Data Understanding & Data Preparation
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from google.cloud import storage


# Dataset sources
DATASETS = {
    "mental_health_longparas": {
        "name": "phoenix1803/Mental-Health-LongParas",
        "split": "train",
        "text_column": "text",
        "has_labels": True,
    },
    "counseling_conversations": {
        "name": "Amod/mental_health_counseling_conversations",
        "split": "train",
        "text_column": "Context",
        "has_labels": False,
    },
}

GCS_BUCKET = "bloom-ml-data"
GCS_RAW_PATH = "raw"


def download_dataset(dataset_config: dict) -> pd.DataFrame:
    """Download a single dataset from Hugging Face."""
    print(f"Downloading {dataset_config['name']}...")

    try:
        ds = load_dataset(dataset_config["name"], split=dataset_config["split"])
        df = ds.to_pandas()
        print(f"  Downloaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"  Error downloading {dataset_config['name']}: {e}")
        return pd.DataFrame()


def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = GCS_BUCKET):
    """Upload file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    blob.upload_from_filename(local_path)
    print(f"Uploaded to gs://{bucket_name}/{gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="Download mental health datasets")
    parser.add_argument("--output-dir", type=str, default="./data/raw",
                        help="Local output directory")
    parser.add_argument("--upload-gcs", action="store_true",
                        help="Upload to GCS after download")
    parser.add_argument("--version", type=str, default="v1",
                        help="Dataset version for GCS path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dataframes = {}

    # Download each dataset
    for key, config in DATASETS.items():
        df = download_dataset(config)
        if not df.empty:
            # Save individual dataset
            local_path = output_dir / f"{key}.parquet"
            df.to_parquet(local_path, index=False)
            print(f"  Saved to {local_path}")

            all_dataframes[key] = df

            if args.upload_gcs:
                gcs_path = f"{GCS_RAW_PATH}/{args.version}/{key}.parquet"
                upload_to_gcs(str(local_path), gcs_path)

    # Create combined dataset (primary training data with labels)
    if "mental_health_longparas" in all_dataframes:
        primary_df = all_dataframes["mental_health_longparas"]
        combined_path = output_dir / "combined_training.parquet"
        primary_df.to_parquet(combined_path, index=False)
        print(f"\nCombined training data saved to {combined_path}")
        print(f"Total rows: {len(primary_df)}")

        if args.upload_gcs:
            gcs_path = f"{GCS_RAW_PATH}/{args.version}/combined_training.parquet"
            upload_to_gcs(str(combined_path), gcs_path)

    print("\nDataset download complete!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
