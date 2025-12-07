/**
 * k6 Load Test: WebSocket Connections
 *
 * Tests Socket.io WebSocket connections for real-time messaging.
 * Simulates multiple users connecting and exchanging messages.
 *
 * Run: k6 run load-tests/websocket-load.js
 */

import { check, sleep } from 'k6';
import ws from 'k6/ws';
import { Rate, Counter, Trend } from 'k6/metrics';

// Custom metrics
const connectionErrors = new Rate('ws_connection_errors');
const messageLatency = new Trend('ws_message_latency');
const messagesReceived = new Counter('ws_messages_received');
const connectionDuration = new Trend('ws_connection_duration');

export const options = {
  stages: [
    { duration: '30s', target: 20 },   // Ramp up to 20 WebSocket connections
    { duration: '1m', target: 50 },    // Ramp up to 50 connections
    { duration: '2m', target: 100 },   // Stay at 100 connections
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    ws_connection_errors: ['rate<0.1'],
    ws_message_latency: ['p(95)<500'],
  },
};

// Environment
const WS_URL = __ENV.WS_URL || 'ws://localhost:3001';

export default function () {
  const url = `${WS_URL}/socket.io/?EIO=4&transport=websocket`;
  const startTime = Date.now();

  const res = ws.connect(url, {}, function (socket) {
    let connected = false;
    let messageStart = 0;

    socket.on('open', () => {
      connected = true;
      console.log(`WebSocket connected: VU ${__VU}`);

      // Socket.io handshake (send probe)
      socket.send('40');
    });

    socket.on('message', (data) => {
      messagesReceived.add(1);

      // Handle Socket.io protocol messages
      if (data === '2') {
        // Ping - respond with pong
        socket.send('3');
      }

      if (data.startsWith('0')) {
        // Connection established
        console.log('Socket.io handshake complete');
      }

      if (data.startsWith('42')) {
        // Event message
        const latency = Date.now() - messageStart;
        messageLatency.add(latency);
      }
    });

    socket.on('close', () => {
      const duration = Date.now() - startTime;
      connectionDuration.add(duration);
      console.log(`WebSocket closed after ${duration}ms`);
    });

    socket.on('error', (e) => {
      connectionErrors.add(1);
      console.log(`WebSocket error: ${e.error()}`);
    });

    // Simulate user activity
    sleep(Math.random() * 2 + 1);

    // Send a test message (Socket.io format)
    messageStart = Date.now();
    socket.send('42["message",{"type":"ping","timestamp":' + Date.now() + '}]');

    // Keep connection alive
    sleep(5);

    // Close gracefully
    socket.close();
  });

  // Check connection result
  check(res, {
    'WebSocket connection successful': (r) => r && r.status === 101,
  });

  if (!res || res.status !== 101) {
    connectionErrors.add(1);
  }
}

export function setup() {
  console.log('='.repeat(60));
  console.log('Bloom Health WebSocket Load Test');
  console.log('='.repeat(60));
  console.log(`Target: ${WS_URL}`);
  console.log('='.repeat(60));

  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log('='.repeat(60));
  console.log('WebSocket Load Test Complete');
  console.log(`Started: ${data.startTime}`);
  console.log(`Ended: ${new Date().toISOString()}`);
  console.log('='.repeat(60));
}
