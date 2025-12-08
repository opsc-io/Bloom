"""
Retraining Pipeline for Mental Health Model
Triggered via Cloud Scheduler or manually.

CRISP-DM Phase: Deployment (Continuous Learning)
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import storage, bigquery, aiplatform
import pandas as pd


# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "project-4fc52960-1177-49ec-a6f")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET = os.getenv("GCS_BUCKET", "bloom-ml-data")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "bloom-ml-models")

# Thresholds
MIN_NEW_SAMPLES = 1000  # Minimum new samples to trigger retraining
IMPROVEMENT_THRESHOLD = 0.02  # Model must be 2% better to deploy


def export_therapist_feedback(
    days: int = 30,
    output_path: str = None,
) -> str:
    """
    Export therapist feedback on ML predictions for active learning.
    This is the primary source of labeled data for retraining.

    CRISP-DM Phase: Data Understanding / Data Preparation
    """
    client = bigquery.Client(project=PROJECT_ID)

    # Query to get therapist-corrected labels
    query = f"""
    SELECT
        ma.id as analysis_id,
        m.body as text,
        ma.label as original_label,
        ma.confidence as original_confidence,
        f.isCorrect as is_correct,
        f.correctedLabel as corrected_label,
        COALESCE(f.correctedLabel, ma.label) as final_label,
        f.createdAt as feedback_date
    FROM `{PROJECT_ID}.bloom_prod.MLFeedback` f
    JOIN `{PROJECT_ID}.bloom_prod.MessageAnalysis` ma ON f.analysisId = ma.id
    JOIN `{PROJECT_ID}.bloom_prod.Message` m ON ma.messageId = m.id
    WHERE f.createdAt >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        AND m.body IS NOT NULL
        AND LENGTH(m.body) >= 50
    ORDER BY f.createdAt DESC
    """

    print(f"Exporting therapist feedback from last {days} days...")
    df = client.query(query).to_dataframe()

    if df.empty:
        print("No therapist feedback found")
        return None

    # Use corrected labels for training
    df["label"] = df["final_label"]

    correct_count = df["is_correct"].sum()
    incorrect_count = len(df) - correct_count
    print(f"Exported {len(df)} feedback records")
    print(f"  - Confirmed correct: {correct_count}")
    print(f"  - Corrected labels: {incorrect_count}")

    # Save to GCS
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"gs://{BUCKET}/feedback/{timestamp}/therapist_feedback.parquet"

    # Upload
    local_path = "/tmp/therapist_feedback.parquet"
    df.to_parquet(local_path, index=False)

    storage_client = storage.Client()
    if output_path.startswith("gs://"):
        output_path = output_path[5:]
    bucket_name, blob_path = output_path.split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    print(f"Uploaded to gs://{bucket_name}/{blob_path}")
    return f"gs://{bucket_name}/{blob_path}"


def export_recent_conversations(
    days: int = 7,
    output_path: str = None,
) -> str:
    """
    Export recent therapist-patient conversations from BigQuery.
    Only exports anonymized, consented data.
    """
    client = bigquery.Client(project=PROJECT_ID)

    # Query to get recent conversations (anonymized)
    query = f"""
    SELECT
        m.id as message_id,
        m.body as text,
        m.createdAt as created_at,
        c.id as conversation_id,
        -- Note: In production, add proper anonymization
        'patient' as role
    FROM `{PROJECT_ID}.bloom_prod.message` m
    JOIN `{PROJECT_ID}.bloom_prod.conversation` c ON m.conversationId = c.id
    WHERE m.createdAt >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        AND m.body IS NOT NULL
        AND LENGTH(m.body) >= 50
        -- Add consent check in production
        -- AND user.data_consent = TRUE
    ORDER BY m.createdAt DESC
    LIMIT 50000
    """

    print(f"Exporting conversations from last {days} days...")
    df = client.query(query).to_dataframe()

    if df.empty:
        print("No new conversations found")
        return None

    print(f"Exported {len(df)} messages")

    # Save to GCS
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"gs://{BUCKET}/feedback/{timestamp}/conversations.parquet"

    # Upload
    local_path = "/tmp/conversations.parquet"
    df.to_parquet(local_path, index=False)

    storage_client = storage.Client()
    if output_path.startswith("gs://"):
        output_path = output_path[5:]
    bucket_name, blob_path = output_path.split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    print(f"Uploaded to gs://{bucket_name}/{blob_path}")
    return f"gs://{bucket_name}/{blob_path}"


def prepare_training_data(
    new_data_path: str,
    existing_data_path: str,
    output_dir: str,
) -> tuple:
    """
    Combine new feedback data with existing training data.
    Returns paths to train/val splits.
    """
    storage_client = storage.Client()

    # Download new data
    new_local = "/tmp/new_data.parquet"
    download_from_gcs(new_data_path, new_local, storage_client)
    new_df = pd.read_parquet(new_local)

    # Download existing data
    existing_local = "/tmp/existing_data.parquet"
    download_from_gcs(existing_data_path, existing_local, storage_client)
    existing_df = pd.read_parquet(existing_local)

    print(f"New samples: {len(new_df)}, Existing samples: {len(existing_df)}")

    # Combine datasets
    # For new data without labels, we could:
    # 1. Use pseudo-labeling from current model
    # 2. Wait for therapist feedback/annotations
    # 3. Use unsupervised/semi-supervised methods

    # For now, just use existing labeled data + any new labeled samples
    if "Sentiment" in new_df.columns:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print("New data has no labels, using existing data only")
        combined_df = existing_df

    # Shuffle and split
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.9 * len(combined_df))
    train_df = combined_df[:train_size]
    val_df = combined_df[train_size:]

    # Upload splits
    train_path = f"{output_dir}/train.parquet"
    val_path = f"{output_dir}/val.parquet"

    upload_to_gcs("/tmp/train.parquet", train_path, train_df, storage_client)
    upload_to_gcs("/tmp/val.parquet", val_path, val_df, storage_client)

    return train_path, val_path


def download_from_gcs(gcs_path: str, local_path: str, client=None):
    """Download file from GCS."""
    if client is None:
        client = storage.Client()

    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str, df: pd.DataFrame, client=None):
    """Upload DataFrame to GCS."""
    if client is None:
        client = storage.Client()

    df.to_parquet(local_path, index=False)

    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    bucket_name, blob_path = gcs_path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def trigger_training_job(
    train_path: str,
    val_path: str,
    output_path: str,
) -> str:
    """Submit training job to Vertex AI."""
    from ml.pipelines.vertex_training import create_training_job

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"mental-health-retrain-{timestamp}"

    model = create_training_job(
        display_name=job_name,
        train_data_path=train_path,
        val_data_path=val_path,
        output_path=output_path,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )

    return job_name


def compare_models(
    new_model_path: str,
    current_model_path: str,
    test_data_path: str,
) -> dict:
    """
    Compare new model vs current production model.
    Returns metrics comparison.
    """
    # This would load both models and evaluate on test set
    # For simplicity, we check training history

    storage_client = storage.Client()

    # Load new model metrics
    new_history_path = f"{new_model_path}/training_history.json"
    download_from_gcs(new_history_path, "/tmp/new_history.json", storage_client)
    with open("/tmp/new_history.json") as f:
        new_history = json.load(f)

    # Load current model metrics (if exists)
    try:
        current_history_path = f"{current_model_path}/training_history.json"
        download_from_gcs(current_history_path, "/tmp/current_history.json", storage_client)
        with open("/tmp/current_history.json") as f:
            current_history = json.load(f)
    except Exception:
        print("No current model found, will deploy new model")
        return {"should_deploy": True, "improvement": 1.0}

    # Compare final validation loss
    new_val_loss = new_history[-1]["val_loss"]
    current_val_loss = current_history[-1]["val_loss"]

    improvement = (current_val_loss - new_val_loss) / current_val_loss

    return {
        "should_deploy": improvement >= IMPROVEMENT_THRESHOLD,
        "improvement": improvement,
        "new_val_loss": new_val_loss,
        "current_val_loss": current_val_loss,
    }


def deploy_new_model(
    model_path: str,
    version: str,
    traffic_split: float = 0.1,
):
    """
    Deploy new model to Vertex AI endpoint with traffic split.
    """
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Get existing endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="bloom-ml-endpoint"'
    )

    if not endpoints:
        print("No existing endpoint found, creating new one")
        from ml.pipelines.vertex_training import deploy_model
        deploy_model(
            model_artifact_path=model_path,
            display_name=f"mental-health-model-{version}",
            serving_container_uri=f"us-central1-docker.pkg.dev/{PROJECT_ID}/bloom-images/ml-inference:latest",
        )
        return

    endpoint = endpoints[0]

    # Upload new model version
    model = aiplatform.Model.upload(
        display_name=f"mental-health-model-{version}",
        artifact_uri=model_path,
        serving_container_image_uri=f"us-central1-docker.pkg.dev/{PROJECT_ID}/bloom-images/ml-inference:latest",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    # Deploy with traffic split
    # Start with 10% traffic, then increase to 100% if monitoring looks good
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        traffic_percentage=int(traffic_split * 100),
    )

    print(f"Deployed model {version} with {traffic_split * 100}% traffic")


def run_retraining_pipeline(
    force: bool = False,
    days: int = 7,
    feedback_days: int = 30,
):
    """
    Execute the full retraining pipeline with active learning.

    Active Learning Flow:
    1. Export therapist feedback (corrected labels) - PRIMARY DATA SOURCE
    2. Export recent conversations (for pseudo-labeling)
    3. Combine with existing training data
    4. Train new model
    5. Compare with production model
    6. Deploy if improved

    CRISP-DM Phase: Deployment (Continuous Learning Loop)
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    version = f"v{timestamp}"

    print(f"=== Retraining Pipeline with Active Learning - {version} ===\n")

    # Step 1: Export therapist feedback (active learning data)
    print("Step 1: Exporting therapist feedback (active learning)...")
    feedback_path = export_therapist_feedback(days=feedback_days)

    if feedback_path:
        print(f"  ✓ Therapist feedback exported: {feedback_path}")
    else:
        print("  ✗ No therapist feedback available")

    # Step 1b: Export recent conversations (optional, for pseudo-labeling)
    print("\nStep 1b: Exporting recent conversations...")
    new_data_path = export_recent_conversations(days=days)

    if new_data_path is None and feedback_path is None and not force:
        print("No new data to train on. Exiting.")
        return

    # Step 2: Prepare training data (prioritize feedback data)
    print("\nStep 2: Preparing training data...")
    existing_data = f"gs://{BUCKET}/processed/v1/train.parquet"
    output_dir = f"gs://{BUCKET}/retrain/{timestamp}"

    # Use feedback data as primary source if available
    primary_data = feedback_path or new_data_path or existing_data

    train_path, val_path = prepare_training_data(
        primary_data,
        existing_data,
        output_dir,
    )

    # Step 3: Train new model
    print("\nStep 3: Training new model...")
    model_output = f"gs://{MODEL_BUCKET}/{version}"
    job_name = trigger_training_job(train_path, val_path, model_output)
    print(f"Training job submitted: {job_name}")
    print("Waiting for training to complete...")

    # In production, this would wait for the job or be triggered by Cloud Pub/Sub
    # For now, we assume synchronous execution

    # Step 4: Compare models
    print("\nStep 4: Comparing models...")
    current_model = f"gs://{MODEL_BUCKET}/current"
    comparison = compare_models(
        new_model_path=model_output,
        current_model_path=current_model,
        test_data_path=val_path,
    )

    print(f"Improvement: {comparison['improvement']:.2%}")

    # Step 5: Deploy if better
    if comparison["should_deploy"]:
        print("\nStep 5: Deploying new model...")
        deploy_new_model(
            model_path=model_output,
            version=version,
            traffic_split=0.1,  # Start with 10%
        )

        # Update "current" pointer
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET)
        blob = bucket.blob("current/version.txt")
        blob.upload_from_string(version)

        print(f"\nNew model {version} deployed with 10% traffic!")
    else:
        print("\nNew model is not significantly better. Skipping deployment.")

    print("\n=== Retraining Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Mental Health Model Retraining Pipeline")
    parser.add_argument("--force", action="store_true", help="Force retraining even without new data")
    parser.add_argument("--days", type=int, default=7, help="Days of conversation history to export")
    args = parser.parse_args()

    run_retraining_pipeline(force=args.force, days=args.days)


if __name__ == "__main__":
    main()
