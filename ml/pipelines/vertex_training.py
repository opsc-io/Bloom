"""
Vertex AI Training Pipeline Launcher
Submits training jobs to Vertex AI with hyperparameter tuning support.

CRISP-DM Phase: Modeling
"""

import os
import argparse
from datetime import datetime
from google.cloud import aiplatform


# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "project-4fc52960-1177-49ec-a6f")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET = os.getenv("GCS_BUCKET", "bloom-ml-data")

# Training container
TRAINING_CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest"

# Machine configuration
MACHINE_TYPE = "n1-standard-8"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1


def create_training_job(
    display_name: str,
    train_data_path: str,
    val_data_path: str,
    output_path: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """Create and run a Vertex AI custom training job."""

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Training script path (uploaded to GCS)
    training_script = f"gs://{BUCKET}/training/train.py"

    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path="ml/training/train.py",
        container_uri=TRAINING_CONTAINER,
        requirements=[
            "transformers>=4.30.0",
            "datasets>=2.13.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "google-cloud-storage>=2.10.0",
            "cloudml-hypertune>=0.1.0.dev6",
        ],
    )

    # Run the training job
    model = job.run(
        replica_count=1,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        args=[
            f"--train-data={train_data_path}",
            f"--val-data={val_data_path}",
            f"--output-dir=/tmp/model",
            f"--model-gcs={output_path}",
            f"--epochs={epochs}",
            f"--batch-size={batch_size}",
            f"--learning-rate={learning_rate}",
        ],
    )

    return model


def create_hyperparameter_tuning_job(
    display_name: str,
    train_data_path: str,
    val_data_path: str,
    output_path: str,
    max_trial_count: int = 10,
    parallel_trial_count: int = 2,
):
    """Create a hyperparameter tuning job on Vertex AI."""

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Define hyperparameter search space
    hparam_spec = {
        "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=1e-6, max=1e-4, scale="log"
        ),
        "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
            values=[8, 16, 32], scale="linear"
        ),
        "epochs": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
            values=[2, 3, 5], scale="linear"
        ),
    }

    # Create the job
    job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=aiplatform.CustomJob(
            display_name=f"{display_name}-trial",
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": MACHINE_TYPE,
                        "accelerator_type": ACCELERATOR_TYPE,
                        "accelerator_count": ACCELERATOR_COUNT,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": TRAINING_CONTAINER,
                        "package_uris": [f"gs://{BUCKET}/training/trainer.tar.gz"],
                        "python_module": "trainer.train",
                        "args": [
                            f"--train-data={train_data_path}",
                            f"--val-data={val_data_path}",
                            f"--output-dir=/tmp/model",
                            f"--model-gcs={output_path}",
                        ],
                    },
                }
            ],
        ),
        metric_spec={"val_loss": "minimize"},
        parameter_spec=hparam_spec,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    job.run()
    return job


def deploy_model(
    model_artifact_path: str,
    display_name: str,
    serving_container_uri: str,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 2,
):
    """Deploy a trained model to a Vertex AI endpoint."""

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Upload model to Model Registry
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_artifact_path,
        serving_container_image_uri=serving_container_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=f"{display_name}-endpoint")

    # Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=1,
    )

    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training command
    train_parser = subparsers.add_parser("train", help="Run training job")
    train_parser.add_argument("--train-data", required=True, help="GCS path to training data")
    train_parser.add_argument("--val-data", required=True, help="GCS path to validation data")
    train_parser.add_argument("--output", required=True, help="GCS path for model output")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument("--train-data", required=True)
    tune_parser.add_argument("--val-data", required=True)
    tune_parser.add_argument("--output", required=True)
    tune_parser.add_argument("--max-trials", type=int, default=10)
    tune_parser.add_argument("--parallel-trials", type=int, default=2)

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model to endpoint")
    deploy_parser.add_argument("--model-path", required=True, help="GCS path to model artifacts")
    deploy_parser.add_argument("--container", required=True, help="Container image URI")
    deploy_parser.add_argument("--name", required=True, help="Display name for model")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.command == "train":
        job_name = f"mental-health-training-{timestamp}"
        create_training_job(
            display_name=job_name,
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(f"Training job {job_name} submitted successfully!")

    elif args.command == "tune":
        job_name = f"mental-health-hpt-{timestamp}"
        create_hyperparameter_tuning_job(
            display_name=job_name,
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_path=args.output,
            max_trial_count=args.max_trials,
            parallel_trial_count=args.parallel_trials,
        )
        print(f"Hyperparameter tuning job {job_name} submitted successfully!")

    elif args.command == "deploy":
        endpoint = deploy_model(
            model_artifact_path=args.model_path,
            display_name=args.name,
            serving_container_uri=args.container,
        )
        print(f"Model deployed to: {endpoint.resource_name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
