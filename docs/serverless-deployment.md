# Serverless Deployment (Vertex AI)

Bloom uses Vertex AI’s serverless prediction endpoints for ML inference while the web app runs on GKE/Next.js. This keeps the model autoscaled independently of the app pods.

## Architecture
- **Inference container**: `ml-training/inference.py` packaged via `Dockerfile.inference`.
- **Endpoint**: Vertex AI (region `us-central1`), machine type `n1-standard-4`, autoscale `min=1`, `max=2` (keeps one warm pod to avoid cold starts).
- **Identity**: Workload Identity → K8s SA (`bloom-sa`) impersonates GCP SA (`bloom-app-sa`) with `roles/aiplatform.user`.
- **Network**: Egress over Google network; optional VPC-SC per project. No public inbound ports; only the Vertex endpoint HTTPS URL is exposed.

## Build & Deploy
```bash
# Build inference image (from repo root)
cd ml-training
docker buildx build \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-inference:latest \
  -f Dockerfile.inference \
  --push .

# Deploy to Vertex AI
gcloud ai models upload \
  --region=us-central1 \
  --display-name=bloom-mental-health-v2 \
  --container-image-uri=us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/ml-inference:latest

gcloud ai endpoints create --region=us-central1 --display-name=bloom-mental-health
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=2
```

## App Integration
- **Client**: `src/lib/ml-inference.ts` posts to the Vertex endpoint URL set in `ML_ENDPOINT_URL`.
- **Secrets**: No API keys; auth is via GCP token fetched from metadata server when running in GKE with Workload Identity.
- **Timeouts/retries**: Use short timeouts (2–5s) and retry on `429/503` with exponential backoff; if `USE_MOCK_ML=true` the Next.js API falls back to deterministic mock scores.

## CI/CD Considerations
- Build and push the inference image in GitHub Actions before deploying manifests.
- Keep model artifacts in GCS (`bloom-ml-training/models/v2`) and point the inference image at the mounted weights or GCS download on startup.
- Vertex IAM: restrict `aiplatform.endpointInvoker` to the GKE workload identity SA and CI deployer roles; audit logs kept in Cloud Logging.

## Rollback & Canary
- Deploy new model versions to the same endpoint with `--traffic-split=0=x,1=y` for canaries.
- If latency spikes, temporarily raise `min-replica-count` or scale down `batching` in the FastAPI handler.
