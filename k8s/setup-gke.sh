#!/bin/bash
set -euo pipefail

# ============================================
# Bloom Health - GKE Setup Script
# ============================================
# This script sets up a production-ready GKE cluster
# and deploys the Bloom application
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - kubectl installed
#   - Docker installed (for building images)
#
# Usage:
#   ./k8s/setup-gke.sh
# ============================================

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
CLUSTER_NAME="${CLUSTER_NAME:-bloom-cluster}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
DOMAIN="${DOMAIN:-bloom.example.com}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Install with: gcloud components install kubectl"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Install from https://docs.docker.com/get-docker/"
        exit 1
    fi

    log_info "All prerequisites met!"
}

# Set up GCP project
setup_project() {
    log_info "Setting up GCP project: $PROJECT_ID"

    gcloud config set project "$PROJECT_ID"

    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable \
        container.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        secretmanager.googleapis.com \
        --quiet
}

# Create GKE Autopilot cluster
create_cluster() {
    log_info "Creating GKE Autopilot cluster: $CLUSTER_NAME"

    if gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" &> /dev/null; then
        log_warn "Cluster $CLUSTER_NAME already exists, skipping creation"
    else
        gcloud container clusters create-auto "$CLUSTER_NAME" \
            --region="$REGION" \
            --release-channel=regular \
            --enable-master-authorized-networks \
            --master-authorized-networks="0.0.0.0/0" \
            --quiet
    fi

    # Get credentials
    log_info "Getting cluster credentials..."
    gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"
}

# Create Artifact Registry for Docker images
setup_registry() {
    log_info "Setting up Artifact Registry..."

    REPO_NAME="bloom-images"

    if gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &> /dev/null; then
        log_warn "Repository $REPO_NAME already exists"
    else
        gcloud artifacts repositories create "$REPO_NAME" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Bloom Docker images"
    fi

    # Configure Docker auth
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."

    IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/bloom-images"

    # Build app image
    log_info "Building bloom-app image..."
    docker build -t "${IMAGE_BASE}/bloom-app:latest" -f Dockerfile .
    docker push "${IMAGE_BASE}/bloom-app:latest"

    # Build socket image
    log_info "Building bloom-socket image..."
    docker build -t "${IMAGE_BASE}/bloom-socket:latest" -f Dockerfile.socket .
    docker push "${IMAGE_BASE}/bloom-socket:latest"

    log_info "Images pushed to Artifact Registry"
}

# Create static IP for Ingress
create_static_ip() {
    log_info "Creating static IP for Ingress..."

    if gcloud compute addresses describe bloom-ip --global &> /dev/null; then
        log_warn "Static IP 'bloom-ip' already exists"
    else
        gcloud compute addresses create bloom-ip --global
    fi

    IP=$(gcloud compute addresses describe bloom-ip --global --format="value(address)")
    log_info "Static IP: $IP"
    log_warn "Point your DNS A record for $DOMAIN to: $IP"
}

# Create secrets in Secret Manager (optional, for better security)
create_secrets() {
    log_info "Creating secrets..."

    # Generate auth secret if not set
    AUTH_SECRET="${BETTER_AUTH_SECRET:-$(openssl rand -base64 32)}"

    log_warn "Creating Kubernetes secrets..."
    log_warn "In production, use GCP Secret Manager with External Secrets Operator"

    # For now, create as K8s secret directly
    kubectl create namespace bloom --dry-run=client -o yaml | kubectl apply -f -

    kubectl create secret generic bloom-secrets \
        --namespace=bloom \
        --from-literal=DATABASE_URL="${DATABASE_URL:-postgresql://root@cockroachdb:26257/bloom?sslmode=disable}" \
        --from-literal=BETTER_AUTH_SECRET="$AUTH_SECRET" \
        --from-literal=BETTER_AUTH_URL="https://${DOMAIN}" \
        --from-literal=NEXT_PUBLIC_SOCKET_URL="https://${DOMAIN}" \
        --from-literal=SMTP_HOST="${SMTP_HOST:-}" \
        --from-literal=SMTP_PORT="${SMTP_PORT:-587}" \
        --from-literal=SMTP_USER="${SMTP_USER:-}" \
        --from-literal=SMTP_PASSWORD="${SMTP_PASSWORD:-}" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "Secrets created. Auth secret: $AUTH_SECRET"
    log_warn "SAVE THIS SECRET! You'll need it for the application."
}

# Update K8s manifests with actual values
update_manifests() {
    log_info "Updating Kubernetes manifests..."

    IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/bloom-images"

    # Update image references in deployments
    sed -i.bak "s|gcr.io/PROJECT_ID/bloom-app:latest|${IMAGE_BASE}/bloom-app:latest|g" k8s/base/app-deployment.yaml
    sed -i.bak "s|gcr.io/PROJECT_ID/bloom-socket:latest|${IMAGE_BASE}/bloom-socket:latest|g" k8s/base/socket-deployment.yaml

    # Update domain in ingress
    sed -i.bak "s|bloom.example.com|${DOMAIN}|g" k8s/base/ingress.yaml
    sed -i.bak "s|bloom.example.com|${DOMAIN}|g" k8s/base/socket-deployment.yaml

    # Clean up backup files
    rm -f k8s/base/*.bak

    log_info "Manifests updated"
}

# Deploy to GKE
deploy() {
    log_info "Deploying to GKE..."

    # Apply all manifests
    kubectl apply -k k8s/base/

    log_info "Waiting for deployments to be ready..."
    kubectl rollout status deployment/bloom-app -n bloom --timeout=300s
    kubectl rollout status deployment/bloom-socket -n bloom --timeout=300s

    log_info "Deployment complete!"
}

# Print status and next steps
print_status() {
    log_info "============================================"
    log_info "Bloom Health GKE Deployment Complete!"
    log_info "============================================"
    echo ""

    IP=$(gcloud compute addresses describe bloom-ip --global --format="value(address)" 2>/dev/null || echo "pending")

    echo "Static IP: $IP"
    echo "Domain: $DOMAIN"
    echo ""
    echo "Next steps:"
    echo "1. Point DNS A record for $DOMAIN to $IP"
    echo "2. Wait for SSL certificate provisioning (can take 15-60 minutes)"
    echo "3. Set up CockroachDB (Cloud or self-hosted)"
    echo "4. Update DATABASE_URL in secrets if needed"
    echo ""
    echo "Useful commands:"
    echo "  kubectl get pods -n bloom"
    echo "  kubectl logs -f deployment/bloom-app -n bloom"
    echo "  kubectl get ingress -n bloom"
    echo "  kubectl describe managedcertificate bloom-certificate -n bloom"
}

# Main
main() {
    echo "============================================"
    echo "Bloom Health - GKE Setup"
    echo "============================================"
    echo ""

    if [[ "$PROJECT_ID" == "your-project-id" ]]; then
        log_error "Please set GCP_PROJECT_ID environment variable"
        echo "  export GCP_PROJECT_ID=your-actual-project-id"
        echo "  export DOMAIN=your-domain.com"
        exit 1
    fi

    check_prerequisites
    setup_project
    create_cluster
    setup_registry
    build_and_push_images
    create_static_ip
    update_manifests
    create_secrets
    deploy
    print_status
}

# Run main function
main "$@"
