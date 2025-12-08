/**
 * k6 Load Test: Health Check Endpoint
 *
 * Tests the /api/health endpoint under various load conditions.
 * This is a baseline test to verify infrastructure stability.
 *
 * Run: k6 run load-tests/health-check.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const healthCheckDuration = new Trend('health_check_duration');

// Test configuration
export const options = {
  // Staged load test
  stages: [
    { duration: '30s', target: 10 },   // Ramp up to 10 users
    { duration: '1m', target: 50 },    // Ramp up to 50 users
    { duration: '2m', target: 100 },   // Stay at 100 users
    { duration: '30s', target: 0 },    // Ramp down
  ],

  // Thresholds for pass/fail
  thresholds: {
    http_req_duration: ['p(95)<500'],   // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],      // Less than 1% failures
    errors: ['rate<0.05'],               // Less than 5% error rate
  },
};

// Environment configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  // Health check request
  const healthRes = http.get(`${BASE_URL}/api/health`, {
    headers: {
      'Content-Type': 'application/json',
    },
    tags: { name: 'HealthCheck' },
  });

  // Track custom metrics
  healthCheckDuration.add(healthRes.timings.duration);

  // Validate response
  const success = check(healthRes, {
    'status is 200 or 503': (r) => r.status === 200 || r.status === 503,
    'response has status field': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status !== undefined;
      } catch {
        return false;
      }
    },
    'response has checks object': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.checks !== undefined;
      } catch {
        return false;
      }
    },
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  // Record errors
  errorRate.add(!success);

  // Think time between requests
  sleep(Math.random() * 2 + 1);
}

// Lifecycle hooks
export function setup() {
  console.log(`Starting load test against: ${BASE_URL}`);

  // Verify target is reachable
  const res = http.get(`${BASE_URL}/api/health`);
  if (res.status !== 200 && res.status !== 503) {
    throw new Error(`Target not reachable: ${res.status}`);
  }

  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log(`Load test completed. Started at: ${data.startTime}`);
}
