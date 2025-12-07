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

| Environment | Domain | Cluster | Namespace |
|-------------|--------|---------|-----------|
| DEV | dev.gcp.bloomhealth.us | bloom-dev-cluster | bloom-dev |
| QA | qa.gcp.bloomhealth.us | bloom-qa-cluster | bloom-qa |
| PROD | bloomhealth.us | bloom-prod-cluster | bloom-prod |

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
# Connect to cluster
gcloud container clusters get-credentials <cluster-name> --zone <zone>

# Deploy using kustomize
kubectl apply -k k8s/overlays/<env>

# Or build and apply
kustomize build k8s/overlays/<env> | kubectl apply -f -
```

### CI/CD

Deployments are automated via GitHub Actions:
- **QA**: Push to `qa` branch triggers `.github/workflows/deploy-qa.yml`
- **PROD**: Push to `main` branch triggers `.github/workflows/deploy-prod.yml`

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
