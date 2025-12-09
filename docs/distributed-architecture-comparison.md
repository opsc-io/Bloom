# Distributed Architecture Comparison

## Bloom (Current)
- **Data**: CockroachDB (3-node, serializable) for multi-zone HA; Redis for cache/pub-sub.
- **Compute**: GKE Autopilot, HPA 2–10 pods; Socket.io sidecar for real-time.
- **ML**: Vertex AI serverless endpoint (min=1, max=2) decoupled from app autoscaling.
- **Observability**: Prometheus + Grafana + Loki/Promtail.
- **Strengths**: Survives zone loss without manual failover; consistent transactions; real-time workloads scale independently; open-source observability.
- **Trade-offs**: Higher operational complexity than single-VM/postgres; needs SRE guardrails (backups, capacity, TLS/rotation).

## Alternative A: Single-Region Postgres + Monolith (baseline)
- **Pros**: Simplest ops; lowest fixed cost; well-known tooling.
- **Cons**: Single AZ/region failure takes down app; manual failover; write scaling limited; WebSocket + API share same node under load.
- **Fit**: Very small teams, pilot traffic (<500 DAU), low HA requirements.

## Alternative B: RDS/Cloud SQL Postgres + Redis + Node (non-distributed DB)
- **Pros**: Managed backups/patching; simpler than CockroachDB; still supports Redis pub/sub for messaging.
- **Cons**: AZ failover causes seconds/minutes of downtime; replicas are eventual; sharding needed at higher scale; still one-region by default.
- **Fit**: Medium traffic with tolerance for brief failovers; teams without CockroachDB expertise.

## Alternative C: Serverless-first (Cloud Run + Firestore/DynamoDB)
- **Pros**: Near-zero idle cost; automatic scaling; no cluster management.
- **Cons**: Eventual consistency; WebSocket support limited (needs managed sockets or polling); cross-collection transactions weaker; cold starts hurt latency-sensitive messaging.
- **Fit**: Bursty, stateless APIs; not ideal for low-latency chat or strict transactional workflows.

## Alternative D: MongoDB/Document DB Cluster
- **Pros**: Flexible schema for message payloads; built-in sharding; managed options available.
- **Cons**: Transaction semantics weaker than Cockroach/Postgres; multi-document ACID is costlier; requires change streams for pub/sub.
- **Fit**: Content-heavy workloads that benefit from schemaless storage; less aligned with relational appointment/claims data.

## Why Distributed (Bloom)
- Regulatory/clinical context demands **durable writes** and **minimal RPO**; CockroachDB gives HA without custom failover scripts.
- **Real-time messaging** benefits from Redis pub/sub + Socket.io workers decoupled from API pods; horizontal scaling keeps p99 latency <100ms.
- **ML inference** offloaded to **Vertex AI serverless** prevents model cold starts from affecting API latency budgets.
- **Cost posture**: Autopilot + open-source monitoring keeps infra cost lower than proprietary APM + large VM overprovisioning needed for single-node HA.
