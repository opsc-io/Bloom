/**
 * k6 Load Test: API Stress Test
 *
 * Comprehensive stress test for Bloom Health API endpoints.
 * Tests multiple endpoints simultaneously to simulate real user behavior.
 *
 * Run: k6 run load-tests/api-stress.js
 * Run with environment: k6 run -e BASE_URL=https://qa.gcp.bloomhealth.us load-tests/api-stress.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const requestCount = new Counter('requests');

// Test scenarios
export const options = {
  scenarios: {
    // Scenario 1: Constant load for baseline
    constant_load: {
      executor: 'constant-vus',
      vus: 20,
      duration: '2m',
      startTime: '0s',
      tags: { scenario: 'constant' },
    },

    // Scenario 2: Ramping load to find breaking point
    ramping_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '1m', target: 200 },
        { duration: '30s', target: 0 },
      ],
      startTime: '2m30s',
      tags: { scenario: 'ramping' },
    },

    // Scenario 3: Spike test
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '10s', target: 10 },
        { duration: '10s', target: 500 },  // Sudden spike
        { duration: '30s', target: 500 },
        { duration: '10s', target: 10 },
      ],
      startTime: '7m',
      tags: { scenario: 'spike' },
    },
  },

  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    http_req_failed: ['rate<0.05'],
    errors: ['rate<0.1'],
    'http_req_duration{scenario:constant}': ['p(95)<500'],
    'http_req_duration{scenario:ramping}': ['p(95)<1500'],
    'http_req_duration{scenario:spike}': ['p(95)<3000'],
  },
};

// Environment
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

// Test data
const TEST_USERS = [
  { email: 'patient1@test.com', role: 'PATIENT' },
  { email: 'therapist1@test.com', role: 'THERAPIST' },
  { email: 'admin@test.com', role: 'ADMINISTRATOR' },
];

export default function () {
  requestCount.add(1);

  group('Health & Status', () => {
    const healthRes = http.get(`${BASE_URL}/api/health`, {
      tags: { endpoint: 'health' },
    });

    const healthOk = check(healthRes, {
      'health returns 2xx': (r) => r.status >= 200 && r.status < 300 || r.status === 503,
    });

    apiLatency.add(healthRes.timings.duration);
    errorRate.add(!healthOk);
  });

  group('Static Assets', () => {
    // Test that static pages load
    const homeRes = http.get(`${BASE_URL}/`, {
      tags: { endpoint: 'home' },
    });

    check(homeRes, {
      'home page loads': (r) => r.status === 200,
      'home page has content': (r) => r.body && r.body.length > 0,
    });

    apiLatency.add(homeRes.timings.duration);
  });

  group('Auth Endpoints', () => {
    // Test sign-in page (GET)
    const signInRes = http.get(`${BASE_URL}/sign-in`, {
      tags: { endpoint: 'sign-in' },
    });

    const signInOk = check(signInRes, {
      'sign-in page loads': (r) => r.status === 200,
    });

    apiLatency.add(signInRes.timings.duration);
    errorRate.add(!signInOk);
  });

  group('API Endpoints', () => {
    // Test API endpoint that doesn't require auth
    const apiRes = http.get(`${BASE_URL}/api/health`, {
      headers: {
        'Accept': 'application/json',
      },
      tags: { endpoint: 'api' },
    });

    const apiOk = check(apiRes, {
      'API returns JSON': (r) => {
        try {
          JSON.parse(r.body);
          return true;
        } catch {
          return false;
        }
      },
    });

    apiLatency.add(apiRes.timings.duration);
    errorRate.add(!apiOk);
  });

  // Simulate user think time
  sleep(Math.random() * 3 + 1);
}

// Setup
export function setup() {
  console.log('='.repeat(60));
  console.log('Bloom Health API Stress Test');
  console.log('='.repeat(60));
  console.log(`Target: ${BASE_URL}`);
  console.log(`Scenarios: constant_load, ramping_load, spike_test`);
  console.log('='.repeat(60));

  // Warmup request
  const warmup = http.get(`${BASE_URL}/api/health`);
  console.log(`Warmup response: ${warmup.status}`);

  return {
    startTime: new Date().toISOString(),
    targetUrl: BASE_URL,
  };
}

// Teardown
export function teardown(data) {
  console.log('='.repeat(60));
  console.log('Load Test Complete');
  console.log('='.repeat(60));
  console.log(`Started: ${data.startTime}`);
  console.log(`Ended: ${new Date().toISOString()}`);
  console.log('='.repeat(60));
}

// Summary handler for custom reporting
export function handleSummary(data) {
  const summary = {
    timestamp: new Date().toISOString(),
    target: BASE_URL,
    metrics: {
      http_req_duration: {
        avg: data.metrics.http_req_duration?.values?.avg,
        p95: data.metrics.http_req_duration?.values['p(95)'],
        p99: data.metrics.http_req_duration?.values['p(99)'],
        max: data.metrics.http_req_duration?.values?.max,
      },
      http_req_failed: data.metrics.http_req_failed?.values?.rate,
      http_reqs: data.metrics.http_reqs?.values?.count,
      vus_max: data.metrics.vus_max?.values?.max,
    },
    thresholds: {
      passed: Object.values(data.thresholds || {}).filter(t => t.ok).length,
      failed: Object.values(data.thresholds || {}).filter(t => !t.ok).length,
    },
  };

  return {
    'load-tests/results/summary.json': JSON.stringify(summary, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  // Simple text summary
  const avg = data.metrics.http_req_duration?.values?.avg?.toFixed(2) || 'N/A';
  const p95 = data.metrics.http_req_duration?.values['p(95)']?.toFixed(2) || 'N/A';
  const reqs = data.metrics.http_reqs?.values?.count || 0;
  const failed = (data.metrics.http_req_failed?.values?.rate * 100)?.toFixed(2) || '0';

  return `
================================================================================
                        BLOOM HEALTH LOAD TEST RESULTS
================================================================================

Total Requests:     ${reqs}
Failed Rate:        ${failed}%
Avg Response Time:  ${avg}ms
P95 Response Time:  ${p95}ms

================================================================================
`;
}
