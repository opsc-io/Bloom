import { NextRequest, NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import prisma from "@/lib/prisma";

const GRAFANA_URL = "https://opscvisuals.grafana.net";

// Get the appropriate token and dashboard based on environment
const getGrafanaConfig = () => {
  const isProduction = process.env.VERCEL_ENV === "production" ||
    process.env.NODE_ENV === "production";

  return {
    token: isProduction
      ? process.env.GRAFANA_TOKEN_PRODUCTION
      : process.env.GRAFANA_TOKEN_QA,
    dashboardUid: isProduction ? "bloom-production" : "bloom-qa",
  };
};

export async function GET(request: NextRequest) {
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

    const { token, dashboardUid } = getGrafanaConfig();

    if (!token) {
      return NextResponse.json(
        { error: "Grafana token not configured" },
        { status: 500 }
      );
    }

    // Get the endpoint from query params
    const searchParams = request.nextUrl.searchParams;
    const endpoint = searchParams.get("endpoint") || "dashboard";

    let grafanaResponse;

    switch (endpoint) {
      case "dashboard":
        // Fetch dashboard metadata
        grafanaResponse = await fetch(
          `${GRAFANA_URL}/api/dashboards/uid/${dashboardUid}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
          }
        );
        break;

      case "panels":
        // Fetch dashboard and extract panel info
        const dashResponse = await fetch(
          `${GRAFANA_URL}/api/dashboards/uid/${dashboardUid}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
          }
        );

        if (!dashResponse.ok) {
          throw new Error("Failed to fetch dashboard");
        }

        const dashData = await dashResponse.json();
        const panels = dashData.dashboard.panels.map((panel: { id: number; title: string; type: string; gridPos: { h: number; w: number; x: number; y: number } }) => ({
          id: panel.id,
          title: panel.title,
          type: panel.type,
          gridPos: panel.gridPos,
        }));

        return NextResponse.json({ panels, dashboardUid });

      case "render":
        // Render a specific panel as PNG (for embedding)
        const panelId = searchParams.get("panelId");
        const width = searchParams.get("width") || "800";
        const height = searchParams.get("height") || "400";
        const from = searchParams.get("from") || "now-24h";
        const to = searchParams.get("to") || "now";

        if (!panelId) {
          return NextResponse.json(
            { error: "panelId is required" },
            { status: 400 }
          );
        }

        const renderUrl = `${GRAFANA_URL}/render/d-solo/${dashboardUid}?orgId=1&panelId=${panelId}&width=${width}&height=${height}&from=${from}&to=${to}`;

        grafanaResponse = await fetch(renderUrl, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!grafanaResponse.ok) {
          throw new Error("Failed to render panel");
        }

        const imageBuffer = await grafanaResponse.arrayBuffer();
        return new NextResponse(imageBuffer, {
          headers: {
            "Content-Type": "image/png",
            "Cache-Control": "no-cache, no-store, must-revalidate",
          },
        });

      case "embed-url":
        // Return the embed URL for iframe (with auth via service account)
        const embedPanelId = searchParams.get("panelId");
        const embedFrom = searchParams.get("from") || "now-24h";
        const embedTo = searchParams.get("to") || "now";

        // For embedded panels, we use the solo panel view
        const embedUrl = embedPanelId
          ? `${GRAFANA_URL}/d-solo/${dashboardUid}?orgId=1&panelId=${embedPanelId}&from=${embedFrom}&to=${embedTo}&theme=dark`
          : `${GRAFANA_URL}/d/${dashboardUid}?orgId=1&from=${embedFrom}&to=${embedTo}&theme=dark&kiosk`;

        return NextResponse.json({ embedUrl, dashboardUid });

      default:
        return NextResponse.json(
          { error: "Invalid endpoint" },
          { status: 400 }
        );
    }

    if (!grafanaResponse.ok) {
      const errorText = await grafanaResponse.text();
      console.error("Grafana API error:", errorText);
      return NextResponse.json(
        { error: "Failed to fetch from Grafana" },
        { status: grafanaResponse.status }
      );
    }

    const data = await grafanaResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Grafana proxy error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
