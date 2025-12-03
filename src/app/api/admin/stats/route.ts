import { NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import prisma from "@/lib/prisma";

export async function GET() {
  try {
    // Verify authentication and admin status
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Check if user is administrator
    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: { administrator: true },
    });

    if (!user?.administrator) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    // Fetch all admin stats in parallel
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const [
      totalUsers,
      therapistCount,
      adminCount,
      newUsersThisWeek,
      activeSessions,
      recentUsers,
      authMethods,
      userGrowth,
    ] = await Promise.all([
      // Total users
      prisma.user.count(),

      // Therapist count
      prisma.user.count({ where: { therapist: true } }),

      // Admin count
      prisma.user.count({ where: { administrator: true } }),

      // New users this week
      prisma.user.count({
        where: { createdAt: { gte: weekAgo } },
      }),

      // Active sessions (not expired)
      prisma.session.count({
        where: { expiresAt: { gt: now } },
      }),

      // Recent signups (last 10)
      prisma.user.findMany({
        select: {
          id: true,
          firstname: true,
          lastname: true,
          email: true,
          createdAt: true,
          therapist: true,
          administrator: true,
        },
        orderBy: { createdAt: "desc" },
        take: 10,
      }),

      // Auth methods breakdown
      prisma.account.groupBy({
        by: ["providerId"],
        _count: { providerId: true },
      }),

      // User growth over last 30 days (grouped by day)
      prisma.user.findMany({
        where: { createdAt: { gte: monthAgo } },
        select: { createdAt: true },
        orderBy: { createdAt: "asc" },
      }),
    ]);

    // Process user growth data into daily counts
    const growthByDay: Record<string, number> = {};
    userGrowth.forEach((user) => {
      const day = user.createdAt.toISOString().split("T")[0];
      growthByDay[day] = (growthByDay[day] || 0) + 1;
    });

    // Fill in missing days with 0
    const growthData: { date: string; users: number }[] = [];
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const dateStr = date.toISOString().split("T")[0];
      growthData.push({
        date: dateStr,
        users: growthByDay[dateStr] || 0,
      });
    }

    // Process auth methods
    const authMethodsData = authMethods.map((m) => ({
      method: m.providerId === "google" ? "Google" : m.providerId === "credential" ? "Email" : m.providerId,
      count: m._count.providerId,
    }));

    return NextResponse.json({
      overview: {
        totalUsers,
        therapistCount,
        adminCount,
        patientCount: totalUsers - therapistCount - adminCount,
        newUsersThisWeek,
        activeSessions,
      },
      recentUsers,
      authMethods: authMethodsData,
      userGrowth: growthData,
    });
  } catch (error) {
    console.error("Admin stats error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
