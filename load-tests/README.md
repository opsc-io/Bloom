# Bloom Health Load Testing Suite

Performance and stress testing for the Bloom Health platform using [k6](https://k6.io/).

## Overview

This suite contains load tests for:
- **Health Check** - Baseline infrastructure test
- **API Stress** - Comprehensive multi-endpoint stress test
- **WebSocket** - Real-time messaging load test

## Prerequisites

Install k6:

```bash
# macOS
brew install k6

# Windows
choco install k6

# Docker
docker pull grafana/k6
```

## Test Scripts

### 1. Health Check (`health-check.js`)

Basic load test for the `/api/health` endpoint.

```bash
# Local development
k6 run load-tests/health-check.js

# Against QA environment
k6 run -e BASE_URL=https://qa.gcp.bloomhealth.us load-tests/health-check.js

# Against Production
k6 run -e BASE_URL=https://bloomhealth.us load-tests/health-check.js
```

**Scenarios:**
- Ramp up: 0 → 10 → 50 → 100 users
- Duration: ~4 minutes
- Thresholds: p95 < 500ms, error rate < 1%

### 2. API Stress Test (`api-stress.js`)

Comprehensive stress test with multiple scenarios.

```bash
# Run full stress test
k6 run load-tests/api-stress.js

# Run with custom VUs
k6 run --vus 50 --duration 5m load-tests/api-stress.js
```

**Scenarios:**
1. **Constant Load** - 20 VUs for 2 minutes (baseline)
2. **Ramping Load** - 0 → 50 → 100 → 200 VUs (find limits)
3. **Spike Test** - 10 → 500 users sudden spike (resilience)

**Total Duration:** ~8 minutes

### 3. WebSocket Load (`websocket-load.js`)

Tests Socket.io WebSocket connections.

```bash
# Test local Socket.io server
k6 run -e WS_URL=ws://localhost:3001 load-tests/websocket-load.js

# Test deployed environment
k6 run -e WS_URL=wss://socket.gcp.bloomhealth.us load-tests/websocket-load.js
```

**Metrics:**
- Connection success rate
- Message latency
- Concurrent connections

## Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `http_req_duration` | p95 < 500ms | 95th percentile response time |
| `http_req_failed` | rate < 5% | Request failure rate |
| `ws_message_latency` | p95 < 500ms | WebSocket message round-trip |
| `errors` | rate < 10% | Custom error rate |

## Running Tests

### Local Development

```bash
# Start the app
npm run dev

# In another terminal
k6 run load-tests/health-check.js
```

### Against Deployed Environments

```bash
# QA Environment
k6 run -e BASE_URL=https://qa.gcp.bloomhealth.us load-tests/api-stress.js

# Production (use with caution!)
k6 run -e BASE_URL=https://bloomhealth.us -e PROD=true load-tests/health-check.js
```

### With Docker

```bash
docker run -i grafana/k6 run - <load-tests/health-check.js
```

## Output & Reporting

### Console Output

k6 provides real-time metrics in the console:
- Response times (min, med, avg, max, p90, p95)
- Request rate
- Data transferred
- Virtual user count

### JSON Report

```bash
k6 run --out json=results.json load-tests/api-stress.js
```

### InfluxDB + Grafana (Advanced)

```bash
# Send metrics to InfluxDB
k6 run --out influxdb=http://localhost:8086/k6 load-tests/api-stress.js
```

## Test Scenarios Summary

| Test | Max VUs | Duration | Purpose |
|------|---------|----------|---------|
| Health Check | 100 | 4 min | Baseline infrastructure |
| API Stress | 500 | 8 min | Find breaking points |
| WebSocket | 100 | 4 min | Real-time messaging |

## Expected Results

### Production SLO Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Availability | 99.9% | 99.5% |
| p95 Latency | < 200ms | < 500ms |
| p99 Latency | < 500ms | < 1000ms |
| Error Rate | < 0.1% | < 1% |

### Scaling Behavior

| VUs | Expected p95 | Notes |
|-----|--------------|-------|
| 10 | ~50ms | Normal operation |
| 50 | ~100ms | Expected increase |
| 100 | ~200ms | HPA should trigger |
| 200 | ~500ms | Near capacity |
| 500 | ~1000ms+ | May trigger throttling |

## Troubleshooting

### High Error Rates

1. Check pod health: `kubectl get pods -n bloom-{env}`
2. Check HPA: `kubectl get hpa -n bloom-{env}`
3. Check logs: `kubectl logs -l app=bloom-app -n bloom-{env}`

### High Latency

1. Check database connections: CockroachDB dashboard
2. Check Redis: `redis-cli INFO stats`
3. Check network: Pod-to-service latency

### Connection Refused

1. Verify ingress: `kubectl get ingress -n bloom-{env}`
2. Check SSL cert: `curl -v https://qa.gcp.bloomhealth.us/api/health`
3. Verify DNS: `nslookup qa.gcp.bloomhealth.us`

## CI/CD Integration

Add to GitHub Actions:

```yaml
- name: Run Load Tests
  uses: grafana/k6-action@v0.3.1
  with:
    filename: load-tests/health-check.js
    flags: --out json=results.json
  env:
    BASE_URL: ${{ secrets.QA_URL }}
```

## Contributing

1. Keep tests focused and maintainable
2. Add thresholds for all critical metrics
3. Document expected behavior
4. Test locally before committing
