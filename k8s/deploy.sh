#!/bin/bash
set -euo pipefail

# ============================================
# Bloom Health - GKE Deployment Script
# ============================================
# Deploys to QA (minimal) or Production (full HA)
#
# Usage:
#   ./k8s/deploy.sh qa          # Deploy to QA
#   ./k8s/deploy.sh production  # Deploy to Production
#   ./k8s/deploy.sh qa setup    # Full setup including cluster
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

usage() {
    echo "Usage: $0 <environment> [command]"
    echo ""
    echo "Environments:"
    echo "  qa          - Minimal setup (1 replica each, ~\$25-35/month)"
    echo "  production  - Full HA setup (3+ replicas, Redis Sentinel, ~\$150-300/month)"
    echo ""
    echo "Commands:"
    echo "  deploy      - Deploy application (default)"
    echo "  setup       - Full setup including GKE cluster creation"
    echo "  destroy     - Tear down the environment"
    echo "  status      - Show deployment status"
    echo ""
    echo "Examples:"
    echo "  $0 qa                    # Deploy to QA"
    echo "  $0 production setup      # Setup production from scratch"
    echo "  $0 qa status             # Check QA status"
    exit 1
}

# Validate environment
validate_env() {
    local env=$1

    if [[ -z "$PROJECT_ID" ]]; then
        log_error "GCP_PROJECT_ID environment variable is required"
        echo "  export GCP_PROJECT_ID=your-project-id"
        exit 1
    fi

    case $env in
        qa)
            CLUSTER_NAME="bloom-qa-cluster"
            NAMESPACE="bloom-qa"
            DOMAIN="${QA_DOMAIN:-qa.bloom.example.com}"
            STATIC_IP_NAME="bloom-qa-ip"
            MACHINE_TYPE="e2-small"
            NODE_COUNT=1
            ;;
        production)
            CLUSTER_NAME="bloom-prod-cluster"
            NAMESPACE="bloom-prod"
            DOMAIN="${PROD_DOMAIN:-bloom.example.com}"
            STATIC_IP_NAME="bloom-prod-ip"
            MACHINE_TYPE="e2-standard-2"
            NODE_COUNT=3
            ;;
        *)
            log_error "Invalid environment: $env"
            usage
            ;;
    esac

    log_info "Environment: $env"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Namespace: $NAMESPACE"
    log_info "Domain: $DOMAIN"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."

    local missing=()

    command -v gcloud &>/dev/null || missing+=("gcloud")
    command -v kubectl &>/dev/null || missing+=("kubectl")
    command -v docker &>/dev/null || missing+=("docker")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    log_info "All prerequisites met"
}

# Setup GCP project and APIs
setup_project() {
    log_step "Setting up GCP project..."

    gcloud config set project "$PROJECT_ID"

    gcloud services enable \
        container.googleapis.com \
        artifactregistry.googleapis.com \
        secretmanager.googleapis.com \
        --quiet

    log_info "GCP APIs enabled"
}

# Create or get cluster
setup_cluster() {
    local env=$1
    log_step "Setting up GKE cluster: $CLUSTER_NAME"

    if gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" &>/dev/null; then
        log_warn "Cluster already exists, getting credentials..."
    else
        log_info "Creating cluster..."

        if [[ "$env" == "qa" ]]; then
            # QA: Simple Autopilot cluster
            gcloud container clusters create-auto "$CLUSTER_NAME" \
                --region="$REGION" \
                --release-channel=regular \
                --quiet
        else
            # Production: Regional cluster with node pools
            gcloud container clusters create "$CLUSTER_NAME" \
                --region="$REGION" \
                --num-nodes=1 \
                --machine-type="$MACHINE_TYPE" \
                --enable-autoscaling \
                --min-nodes=1 \
                --max-nodes=5 \
                --enable-autorepair \
                --enable-autoupgrade \
                --release-channel=regular \
                --quiet
        fi
    fi

    gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"
    log_info "Cluster ready"
}

# Setup Artifact Registry
setup_registry() {
    log_step "Setting up Artifact Registry..."

    local repo_name="bloom-images"

    if gcloud artifacts repositories describe "$repo_name" --location="$REGION" &>/dev/null; then
        log_warn "Repository already exists"
    else
        gcloud artifacts repositories create "$repo_name" \
            --repository-format=docker \
            --location="$REGION" \
            --quiet
    fi

    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    log_info "Registry ready"
}

# Build and push images
build_images() {
    log_step "Building and pushing Docker images..."

    local image_base="${REGION}-docker.pkg.dev/${PROJECT_ID}/bloom-images"

    cd "$PROJECT_ROOT"

    docker build -t "${image_base}/bloom-app:latest" -f Dockerfile .
    docker push "${image_base}/bloom-app:latest"

    docker build -t "${image_base}/bloom-socket:latest" -f Dockerfile.socket .
    docker push "${image_base}/bloom-socket:latest"

    log_info "Images pushed"
}

# Create static IP
create_static_ip() {
    log_step "Creating static IP: $STATIC_IP_NAME"

    if gcloud compute addresses describe "$STATIC_IP_NAME" --global &>/dev/null; then
        log_warn "Static IP already exists"
    else
        gcloud compute addresses create "$STATIC_IP_NAME" --global
    fi

    local ip=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
    log_info "Static IP: $ip"
    log_warn "Point DNS A record for $DOMAIN to: $ip"
}

# Create secrets
create_secrets() {
    local env=$1
    log_step "Creating Kubernetes secrets..."

    local auth_secret="${BETTER_AUTH_SECRET:-$(openssl rand -base64 32)}"
    local db_url="${DATABASE_URL:-postgresql://root@cockroachdb:26257/bloom?sslmode=disable}"

    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    kubectl create secret generic bloom-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=DATABASE_URL="$db_url" \
        --from-literal=BETTER_AUTH_SECRET="$auth_secret" \
        --from-literal=BETTER_AUTH_URL="https://${DOMAIN}" \
        --from-literal=NEXT_PUBLIC_SOCKET_URL="https://${DOMAIN}" \
        --from-literal=REDIS_URL="redis://redis:6379" \
        --from-literal=SMTP_HOST="${SMTP_HOST:-}" \
        --from-literal=SMTP_PORT="${SMTP_PORT:-587}" \
        --from-literal=SMTP_USER="${SMTP_USER:-}" \
        --from-literal=SMTP_PASSWORD="${SMTP_PASSWORD:-}" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "Secrets created"
    log_warn "BETTER_AUTH_SECRET: $auth_secret"
}

# Update manifests with actual values
update_manifests() {
    local env=$1
    log_step "Updating manifests..."

    local image_base="${REGION}-docker.pkg.dev/${PROJECT_ID}/bloom-images"
    local overlay_dir="$SCRIPT_DIR/overlays/$env"

    # Update image references
    find "$SCRIPT_DIR/base" -name "*.yaml" -exec sed -i.bak \
        "s|gcr.io/PROJECT_ID|${image_base}|g" {} \;

    # Update domain
    find "$overlay_dir" -name "*.yaml" -exec sed -i.bak \
        "s|bloom.example.com|${DOMAIN}|g" {} \;
    find "$overlay_dir" -name "*.yaml" -exec sed -i.bak \
        "s|qa.bloom.example.com|${DOMAIN}|g" {} \;

    # Clean up backups
    find "$SCRIPT_DIR" -name "*.bak" -delete

    log_info "Manifests updated"
}

# Deploy with Kustomize
deploy() {
    local env=$1
    log_step "Deploying to $env..."

    kubectl apply -k "$SCRIPT_DIR/overlays/$env"

    log_info "Waiting for deployments..."
    kubectl rollout status deployment/bloom-app -n "$NAMESPACE" --timeout=300s || true
    kubectl rollout status deployment/bloom-socket -n "$NAMESPACE" --timeout=300s || true

    log_info "Deployment complete!"
}

# Show status
show_status() {
    log_step "Deployment Status"

    echo ""
    echo "=== Pods ==="
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "Namespace not found"

    echo ""
    echo "=== Services ==="
    kubectl get svc -n "$NAMESPACE" 2>/dev/null || echo "Namespace not found"

    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "Namespace not found"

    echo ""
    echo "=== HPA ==="
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || echo "No HPA found"

    if gcloud compute addresses describe "$STATIC_IP_NAME" --global &>/dev/null; then
        local ip=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
        echo ""
        echo "=== Access ==="
        echo "Static IP: $ip"
        echo "URL: https://$DOMAIN"
    fi
}

# Destroy environment
destroy() {
    local env=$1
    log_warn "This will destroy the $env environment!"
    read -p "Are you sure? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log_info "Aborted"
        exit 0
    fi

    log_step "Destroying $env environment..."

    kubectl delete namespace "$NAMESPACE" --ignore-not-found

    read -p "Delete cluster too? (yes/no): " delete_cluster
    if [[ "$delete_cluster" == "yes" ]]; then
        gcloud container clusters delete "$CLUSTER_NAME" --region="$REGION" --quiet
    fi

    log_info "Environment destroyed"
}

# Main
main() {
    local env="${1:-}"
    local command="${2:-deploy}"

    if [[ -z "$env" ]]; then
        usage
    fi

    validate_env "$env"
    check_prerequisites

    case $command in
        setup)
            setup_project
            setup_cluster "$env"
            setup_registry
            build_images
            create_static_ip
            create_secrets "$env"
            update_manifests "$env"
            deploy "$env"
            show_status
            ;;
        deploy)
            build_images
            update_manifests "$env"
            deploy "$env"
            show_status
            ;;
        status)
            show_status
            ;;
        destroy)
            destroy "$env"
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"
