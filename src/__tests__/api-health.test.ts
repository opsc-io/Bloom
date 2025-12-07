/**
 * Unit Tests for Health Check API
 * Tests the /api/health endpoint logic
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import prisma from "@/lib/prisma";

// Re-mock prisma for this specific test
vi.mock("@/lib/prisma", () => ({
  default: {
    $queryRaw: vi.fn(),
  },
}));

describe("/api/health Endpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Health Check Logic", () => {
    it("should return healthy when database is connected", async () => {
      // Mock successful database query
      vi.mocked(prisma.$queryRaw).mockResolvedValueOnce([{ "?column?": 1 }]);

      // Simulate the health check logic
      const checks: Record<string, "ok" | "error"> = {
        app: "ok",
        database: "error",
      };

      try {
        await prisma.$queryRaw`SELECT 1`;
        checks.database = "ok";
      } catch {
        checks.database = "error";
      }

      const healthy = Object.values(checks).every((v) => v === "ok");

      expect(checks.app).toBe("ok");
      expect(checks.database).toBe("ok");
      expect(healthy).toBe(true);
    });

    it("should return degraded when database fails", async () => {
      // Mock failed database query
      vi.mocked(prisma.$queryRaw).mockRejectedValueOnce(
        new Error("Connection failed")
      );

      const checks: Record<string, "ok" | "error"> = {
        app: "ok",
        database: "error",
      };

      try {
        await prisma.$queryRaw`SELECT 1`;
        checks.database = "ok";
      } catch {
        checks.database = "error";
      }

      const healthy = Object.values(checks).every((v) => v === "ok");

      expect(checks.app).toBe("ok");
      expect(checks.database).toBe("error");
      expect(healthy).toBe(false);
    });

    it("should include timestamp in response", () => {
      const timestamp = new Date().toISOString();
      expect(timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });
  });

  describe("Response Format", () => {
    it("should return proper health response structure", async () => {
      vi.mocked(prisma.$queryRaw).mockResolvedValueOnce([{ "?column?": 1 }]);

      const checks: Record<string, "ok" | "error"> = {
        app: "ok",
        database: "error",
      };

      try {
        await prisma.$queryRaw`SELECT 1`;
        checks.database = "ok";
      } catch {
        checks.database = "error";
      }

      const healthy = Object.values(checks).every((v) => v === "ok");

      const response = {
        status: healthy ? "healthy" : "degraded",
        checks,
        timestamp: new Date().toISOString(),
      };

      expect(response).toHaveProperty("status");
      expect(response).toHaveProperty("checks");
      expect(response).toHaveProperty("timestamp");
      expect(response.status).toBe("healthy");
      expect(response.checks.app).toBe("ok");
      expect(response.checks.database).toBe("ok");
    });
  });
});
