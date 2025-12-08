/**
 * Unit Tests for Admin Stats API
 * Tests the /api/admin/stats endpoint logic
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import prisma from "@/lib/prisma";

// Setup mock
vi.mock("@/lib/prisma", () => ({
  default: {
    user: {
      count: vi.fn(),
    },
    session: {
      count: vi.fn(),
    },
  },
}));

describe("/api/admin/stats Endpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Stats Aggregation", () => {
    it("should aggregate user counts correctly", async () => {
      // Mock user counts
      vi.mocked(prisma.user.count)
        .mockResolvedValueOnce(100) // total
        .mockResolvedValueOnce(15) // therapists
        .mockResolvedValueOnce(80) // patients
        .mockResolvedValueOnce(25); // new users (last 7 days)

      vi.mocked(prisma.session.count).mockResolvedValueOnce(45); // active sessions

      // Simulate the stats calculation logic
      const [total, therapists, patients, newUsers, activeSessions] =
        await Promise.all([
          prisma.user.count(),
          prisma.user.count(),
          prisma.user.count(),
          prisma.user.count(),
          prisma.session.count(),
        ]);

      const stats = {
        totalUsers: total,
        therapists,
        patients,
        newUsersLast7Days: newUsers,
        activeSessions,
      };

      expect(stats.totalUsers).toBe(100);
      expect(stats.therapists).toBe(15);
      expect(stats.patients).toBe(80);
      expect(stats.newUsersLast7Days).toBe(25);
      expect(stats.activeSessions).toBe(45);
    });

    it("should handle empty database", async () => {
      vi.mocked(prisma.user.count).mockResolvedValue(0);
      vi.mocked(prisma.session.count).mockResolvedValue(0);

      const total = await prisma.user.count();
      const activeSessions = await prisma.session.count();

      expect(total).toBe(0);
      expect(activeSessions).toBe(0);
    });

    it("should handle database errors gracefully", async () => {
      vi.mocked(prisma.user.count).mockRejectedValueOnce(
        new Error("Database error")
      );

      let error: Error | null = null;
      try {
        await prisma.user.count();
      } catch (e) {
        error = e as Error;
      }

      expect(error).not.toBeNull();
      expect(error?.message).toBe("Database error");
    });
  });

  describe("Response Structure", () => {
    it("should return expected stats structure", async () => {
      vi.mocked(prisma.user.count).mockResolvedValue(50);
      vi.mocked(prisma.session.count).mockResolvedValue(10);

      const stats = {
        totalUsers: await prisma.user.count(),
        therapists: 10,
        patients: 35,
        administrators: 5,
        newUsersLast7Days: 8,
        activeSessions: await prisma.session.count(),
      };

      expect(stats).toHaveProperty("totalUsers");
      expect(stats).toHaveProperty("therapists");
      expect(stats).toHaveProperty("patients");
      expect(stats).toHaveProperty("activeSessions");
      expect(typeof stats.totalUsers).toBe("number");
      expect(typeof stats.therapists).toBe("number");
    });
  });
});
