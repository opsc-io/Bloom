/**
 * Vitest Setup File
 * Global test configuration and mocks
 */

import { vi } from "vitest";

// Mock environment variables for tests
process.env.DATABASE_URL = "postgresql://test:test@localhost:5432/test";
process.env.BETTER_AUTH_SECRET = "test-secret-key-for-testing";
process.env.ML_INFERENCE_URL = "http://localhost:8080";

// Mock Prisma client
vi.mock("@/lib/prisma", () => ({
  default: {
    $queryRaw: vi.fn(),
    user: {
      findUnique: vi.fn(),
      findMany: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
      count: vi.fn(),
    },
    message: {
      findUnique: vi.fn(),
      findMany: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    },
    appointment: {
      findUnique: vi.fn(),
      findMany: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    },
    session: {
      count: vi.fn(),
    },
    messageAnalysis: {
      create: vi.fn(),
      findMany: vi.fn(),
    },
  },
}));

// Mock Redis client
vi.mock("@/lib/redis", () => ({
  redis: {
    get: vi.fn(),
    set: vi.fn(),
    del: vi.fn(),
    publish: vi.fn(),
  },
  getRedis: vi.fn(() => ({
    get: vi.fn(),
    set: vi.fn(),
    del: vi.fn(),
    publish: vi.fn(),
  })),
}));

// Mock fetch for ML inference tests
global.fetch = vi.fn();
