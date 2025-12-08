# Kubernetes Infrastructure

This directory contains Kubernetes manifests for deploying the Bloom Health application on GKE.

## Directory Structure

```
k8s/
├── base/                    # Base manifests (shared across environments)
│   ├── kustomization.yaml   # Kustomize configuration
│   ├── namespace.yaml       # Namespace definition
│   ├── configmap.yaml       # Application configuration
│   ├── app-deployment.yaml  # Next.js app deployment
│   ├── socket-deployment.yaml  # Socket.io server deployment
│   ├── cockroachdb-statefulset.yaml  # CockroachDB database
│   ├── redis-statefulset.yaml  # Redis cache
│   ├── db-init-job.yaml     # Database initialization job
│   ├── monitoring-stack.yaml  # Prometheus, Loki, Grafana, Promtail
│   ├── ingress.yaml         # GKE Ingress with managed SSL
│   └── grafana-dashboards/  # Pre-configured Grafana dashboards
│       ├── overview.json
│       ├── cockroachdb.json
│       ├── containers.json
│       └── redis.json
└── overlays/                # Environment-specific overrides
    ├── dev/                 # Development environment
    ├── qa/                  # QA environment
    └── prod/                # Production environment
```

## Environments

| Environment | Domain | Cluster | Region | Namespace |
|-------------|--------|---------|--------|-----------|
| DEV | dev.gcp.bloomhealth.us | bloom-dev-autopilot | us-west1 | bloom-dev |
| QA | qa.gcp.bloomhealth.us | bloom-qa-cluster | us-central1 | bloom-qa |
| PROD | bloomhealth.us | bloom-prod-autopilot | us-east1 | bloom-prod |

> **Note**: DEV and PROD use GKE Autopilot clusters for cost-efficiency and simplified operations.

## Components

### Application Stack
- **bloom-app**: Next.js web application (port 3000)
- **bloom-socket**: Socket.io WebSocket server (port 4000)
- **cockroachdb**: CockroachDB database (port 26257, admin UI 8080)
- **redis**: Redis cache (port 6379)

### Monitoring Stack
- **Prometheus**: Metrics collection and storage (port 9090)
- **Loki**: Log aggregation (port 3100)
- **Promtail**: Log collection agent (DaemonSet)
- **Grafana**: Visualization dashboards (port 3000, path `/grafana/`)
- **Redis Exporter**: Redis metrics exporter (port 9121)

## Real-Time Messaging

The application uses Socket.io with Redis pub/sub for real-time features like typing indicators and message delivery.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ bloom-socket│────▶│    Redis    │
│  (Browser)  │◀────│  (Socket.io)│◀────│  (Pub/Sub)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  bloom-app  │
                    │  (Next.js)  │
                    └─────────────┘
```

### Redis Pub/Sub Channels

| Channel Pattern | Purpose |
|-----------------|---------|
| `typing:*` | Typing indicator events |
| `message:*` | New message delivery |
| `reaction:*` | Message reaction updates |

### CORS Configuration

Each environment has its own `SOCKET_CORS_ORIGIN` configured in the kustomize overlay:

| Environment | SOCKET_CORS_ORIGIN |
|-------------|-------------------|
| DEV | `https://dev.gcp.bloomhealth.us` |
| QA | `https://qa.gcp.bloomhealth.us` |
| PROD | `https://bloomhealth.us` |

### Session Affinity

WebSocket connections require session affinity to maintain persistent connections:
- **Service**: Uses `sessionAffinity: ClientIP` with 1-hour timeout
- **BackendConfig**: Configures GKE load balancer for CLIENT_IP affinity
- **Connection Timeout**: 24 hours for long-lived WebSocket connections

## Grafana Dashboards

Pre-configured dashboards are available at `https://<domain>/grafana/`:

1. **Overview**: Application health, request rates, error rates
2. **CockroachDB**: Database metrics, query performance, storage
3. **Runtime Metrics**: Go runtime metrics (memory, goroutines, GC)
4. **Redis**: Cache hit rates, memory usage, connections

Grafana is configured with anonymous access for embedded dashboard panels.

## Deployment

### Manual Deployment

```bash
# Connect to DEV cluster (Autopilot)
gcloud container clusters get-credentials bloom-dev-autopilot --region us-west1

# Connect to QA cluster (Standard)
gcloud container clusters get-credentials bloom-qa-cluster --region us-central1

# Connect to PROD cluster (Autopilot)
gcloud container clusters get-credentials bloom-prod-autopilot --region us-east1

# Deploy using kustomize
kubectl apply -k k8s/overlays/<env>

# Or build and apply
kustomize build k8s/overlays/<env> | kubectl apply -f -

# Force image update and rollout
kubectl set image deployment/bloom-app -n bloom-<env> app=<new-image>
kubectl set image deployment/bloom-socket -n bloom-<env> socket=<new-image>
kubectl rollout restart deployment/bloom-app -n bloom-<env>
kubectl rollout restart deployment/bloom-socket -n bloom-<env>
```

### CI/CD

Deployments are automated via GitHub Actions:
- **QA**: Push to `qa` branch triggers `.github/workflows/deploy-qa.yml`
- **PROD**: Push to `main` branch triggers `.github/workflows/deploy-prod.yml`

### Building Docker Images

Images are stored in Google Artifact Registry:

```bash
# Registry location
REGISTRY=us-central1-docker.pkg.dev/project-4fc52960-1177-49ec-a6f/bloom-images

# Build and push App image
docker buildx build --platform linux/amd64 -t $REGISTRY/bloom-app:<env>-latest --push .

# Build and push Socket image
docker buildx build --platform linux/amd64 -t $REGISTRY/bloom-socket:<env>-latest -f Dockerfile.socket --push .

# Build and push DB-Init image
docker buildx build --platform linux/amd64 -t $REGISTRY/bloom-db-init:<env>-latest -f Dockerfile.db-init --push .
```

> **Note**: Use `--platform linux/amd64` when building on Apple Silicon (M1/M2/M3) Macs.

## Promtail Configuration

Promtail runs as a DaemonSet to collect pod logs and send them to Loki.

### Key Configuration

The Promtail config in `monitoring-stack.yaml` uses Kubernetes service discovery:

```yaml
scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      # CRITICAL: Set __host__ for node matching
      - source_labels: [__meta_kubernetes_pod_node_name]
        target_label: __host__
      # Filter to bloom-* namespaces only
      - source_labels: [__meta_kubernetes_namespace]
        regex: bloom-.*
        action: keep
```

### Important Notes

1. **HOSTNAME env var**: The DaemonSet must set `HOSTNAME` from `spec.nodeName`:
   ```yaml
   env:
     - name: HOSTNAME
       valueFrom:
         fieldRef:
           fieldPath: spec.nodeName
   ```

2. **ClusterRoleBinding namespace**: Each overlay patches the namespace for RBAC:
   ```yaml
   - target:
       kind: ClusterRoleBinding
       name: promtail
     patch: |-
       - op: add
         path: /subjects/0/namespace
         value: bloom-<env>
   ```

3. **Log path format**: GKE pod logs are at:
   ```
   /var/log/pods/<namespace>_<pod-name>_<pod-uid>/<container-name>/*.log
   ```

### Troubleshooting Promtail

If Promtail shows "0/0 ready" targets:

1. Check HOSTNAME env var matches node name:
   ```bash
   kubectl exec -n bloom-<env> <promtail-pod> -- printenv HOSTNAME
   kubectl get pod -n bloom-<env> <promtail-pod> -o jsonpath='{.spec.nodeName}'
   ```

2. Delete and recreate DaemonSet (selectors are immutable):
   ```bash
   kubectl delete daemonset promtail -n bloom-<env>
   kubectl apply -f k8s/base/monitoring-stack.yaml -n bloom-<env>
   ```

3. Check positions file:
   ```bash
   kubectl exec -n bloom-<env> <promtail-pod> -- cat /tmp/positions.yaml
   ```

## Troubleshooting Socket/WebSocket

### Typing Indicator Not Working

1. Check socket pod logs for Redis connection:
   ```bash
   kubectl logs -n bloom-<env> -l app=bloom-socket --tail=50 | grep Redis
   ```

   You should see:
   ```
   [Redis] Publisher connected to: redis://redis:6379
   [Redis] Publisher ready
   [Redis] Subscribed to pattern: typing:*, total subscriptions: 1
   ```

2. Check CORS configuration:
   ```bash
   kubectl get deployment bloom-socket -n bloom-<env> -o jsonpath='{.spec.template.spec.containers[0].env}' | jq
   ```

   Verify `SOCKET_CORS_ORIGIN` matches the domain.

3. Test Redis pub/sub manually:
   ```bash
   # Get redis pod
   REDIS_POD=$(kubectl get pods -n bloom-<env> -l app=redis -o jsonpath='{.items[0].metadata.name}')

   # Subscribe to typing events
   kubectl exec -n bloom-<env> $REDIS_POD -- redis-cli PSUBSCRIBE "typing:*"
   ```

### WebSocket Connection Failures

1. Check BackendConfig is applied:
   ```bash
   kubectl get backendconfig bloom-socket-backend-config -n bloom-<env>
   ```

2. Verify session affinity:
   ```bash
   kubectl get svc bloom-socket -n bloom-<env> -o yaml | grep -A5 sessionAffinity
   ```

3. Check ingress annotations for WebSocket support:
   ```bash
   kubectl get ingress bloom-ingress -n bloom-<env> -o yaml | grep -A10 annotations
   ```

## Secrets

Secrets are managed via GCP Secret Manager and injected during deployment:

- `better-auth-secret`: Authentication secret
- `google-client-id` / `google-client-secret`: Google OAuth
- `zoom-client-id` / `zoom-client-secret` / `zoom-secret-token`: Zoom integration
- `smtp-user` / `smtp-password`: Email configuration

## Scaling

HorizontalPodAutoscalers are configured for bloom-app and bloom-socket:
- **QA**: min 1, max 2 replicas
- **PROD**: min 1, max 3 replicas

## Resource Limits

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| bloom-app (QA) | 50m | 200m | 128Mi | 256Mi |
| bloom-app (PROD) | 100m | 500m | 256Mi | 512Mi |
| bloom-socket | 50m | 200m | 256Mi | 512Mi |
| cockroachdb | 100m | 500m | 512Mi | 1Gi |
| redis | 50m | 200m | 128Mi | 256Mi |
| prometheus | 50m | 200m | 256Mi | 512Mi |
| loki | 50m | 200m | 128Mi | 256Mi |
| promtail | 10m | 100m | 64Mi | 128Mi |
| grafana | 50m | 200m | 128Mi | 256Mi |
