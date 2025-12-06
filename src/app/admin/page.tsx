"use client";

import { AppSidebar } from "@/components/app-sidebar"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";
import { Users, Stethoscope, Sparkles, TrendingUp, Clock, ExternalLink, BarChart3 } from "lucide-react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

type AdminStats = {
  overview: {
    totalUsers: number;
    therapistCount: number;
    adminCount: number;
    patientCount: number;
    newUsersThisWeek: number;
    activeSessions: number;
  };
  recentUsers: Array<{
    id: string;
    firstname: string | null;
    lastname: string | null;
    email: string;
    createdAt: string;
    therapist: boolean;
    administrator: boolean;
  }>;
  userGrowth: Array<{ date: string; users: number }>;
};

const COLORS = ["#60a5fa", "#f472b6"];

export default function AdminPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [grafanaUrl, setGrafanaUrl] = useState('');
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    const hostname = window.location.hostname;
    const isProduction = hostname === 'bloomhealth.us' || hostname === 'www.bloomhealth.us';
    const isQa = hostname === 'qa.bloomhealth.us';

    // Use local Grafana for localhost, Grafana Cloud for deployed environments
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      setGrafanaUrl('http://localhost:3001/d/bloom-overview/bloom-overview');
    } else if (isProduction) {
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-production/bloom-production');
    } else if (isQa) {
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa');
    } else {
      setGrafanaUrl('http://localhost:3001/d/bloom-overview/bloom-overview');
    }
  }, []);

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session?.user) return;

    const fetchStats = async () => {
      try {
        const res = await fetch("/api/admin/stats");
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch {
        // Silently fail
      } finally {
        setStatsLoading(false);
      }
    };

    fetchStats();
  }, [isPending, session]);

  const isAdmin = (session?.user as { administrator?: boolean } | undefined)?.administrator === true;

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;

  if (!isAdmin) {
    return (
      <SidebarProvider>
        <AppSidebar user={user} />
        <SidebarInset>
          <header className="flex h-16 shrink-0 items-center gap-2">
            <div className="flex items-center gap-2 px-4">
              <SidebarTrigger className="-ml-1" />
              <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
            </div>
          </header>
          <div className="flex flex-1 flex-col items-center justify-center p-4">
            <Card className="max-w-md">
              <CardHeader>
                <CardTitle className="text-destructive">Access Denied</CardTitle>
                <CardDescription>
                  You do not have administrator privileges to access this page.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={() => router.push("/dashboard")} variant="outline">
                  Return to Dashboard
                </Button>
              </CardContent>
            </Card>
          </div>
        </SidebarInset>
      </SidebarProvider>
    );
  }

  const pieData = stats ? [
    { name: "Patients", value: stats.overview.patientCount },
    { name: "Admins", value: stats.overview.adminCount },
  ] : [];

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const formatJoinDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  };

  const getRoleBadge = (user: AdminStats["recentUsers"][0]) => {
    if (user.administrator) {
      return <Badge variant="default" className="bg-blue-500">Admin</Badge>;
    }
    if (user.therapist) {
      return <Badge variant="secondary">Therapist</Badge>;
    }
    return <Badge variant="outline">Patient</Badge>;
  };

  return (
    <SidebarProvider>
      <AppSidebar user={user} />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          {/* Stats Cards Row */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {statsLoading ? "..." : stats?.overview.totalUsers ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  +{statsLoading ? "..." : stats?.overview.newUsersThisWeek ?? 0} this week
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Therapists</CardTitle>
                <Stethoscope className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {statsLoading ? "..." : stats?.overview.therapistCount ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Active practitioners
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
                <Sparkles className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {statsLoading ? "..." : stats?.overview.activeSessions ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Currently logged in
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Charts Row */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* User Growth Chart */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">User Growth</CardTitle>
                  <CardDescription>New signups over the last 30 days</CardDescription>
                </div>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="h-[200px]">
                {statsLoading ? (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    Loading...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={stats?.userGrowth ?? []}>
                      <defs>
                        <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 11 }}
                        className="text-muted-foreground"
                      />
                      <YAxis
                        tick={{ fontSize: 11 }}
                        className="text-muted-foreground"
                        allowDecimals={false}
                      />
                      <Tooltip
                        labelFormatter={formatDate}
                        contentStyle={{
                          backgroundColor: 'hsl(var(--card))',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px'
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="users"
                        stroke="#60a5fa"
                        fillOpacity={1}
                        fill="url(#colorUsers)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {/* User Distribution Pie Chart */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">User Distribution</CardTitle>
                  <CardDescription>Breakdown by role</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="h-[200px]">
                {statsLoading ? (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    Loading...
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={70}
                        paddingAngle={2}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                        labelLine={false}
                      >
                        {pieData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Legend />
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recent Signups Table */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base">Recent Signups</CardTitle>
                <CardDescription>Latest users to join the platform</CardDescription>
              </div>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b text-left text-sm text-muted-foreground">
                      <th className="pb-3 font-medium">Name</th>
                      <th className="pb-3 font-medium">Email</th>
                      <th className="pb-3 font-medium">Role</th>
                      <th className="pb-3 font-medium">Joined</th>
                    </tr>
                  </thead>
                  <tbody>
                    {statsLoading ? (
                      <tr>
                        <td colSpan={4} className="py-4 text-center text-muted-foreground">
                          Loading...
                        </td>
                      </tr>
                    ) : stats?.recentUsers.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="py-4 text-center text-muted-foreground">
                          No users yet
                        </td>
                      </tr>
                    ) : (
                      stats?.recentUsers.map((user) => (
                        <tr key={user.id} className="border-b last:border-0">
                          <td className="py-3">
                            {user.firstname || user.lastname
                              ? `${user.firstname ?? ""} ${user.lastname ?? ""}`.trim()
                              : "—"}
                          </td>
                          <td className="py-3 text-muted-foreground">{user.email}</td>
                          <td className="py-3">{getRoleBadge(user)}</td>
                          <td className="py-3 text-muted-foreground">
                            {formatJoinDate(user.createdAt)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Grafana Analytics Section */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base">System Monitoring</CardTitle>
                <CardDescription>Live metrics and logs from Grafana</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Open Grafana <ExternalLink className="ml-1 h-3 w-3" />
                </a>
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              {grafanaUrl ? (
                <>
                  {/* Log Volume Chart */}
                  <div className="rounded-lg overflow-hidden border">
                    <iframe
                      src={`${grafanaUrl.split('/d/')[0]}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=4&theme=dark`}
                      width="100%"
                      height="200"
                      frameBorder="0"
                      title="Log Volume by Container"
                    />
                  </div>

                  {/* Errors & Warnings Chart */}
                  <div className="rounded-lg overflow-hidden border">
                    <iframe
                      src={`${grafanaUrl.split('/d/')[0]}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=5&theme=dark`}
                      width="100%"
                      height="200"
                      frameBorder="0"
                      title="Errors and Warnings"
                    />
                  </div>

                  {/* Application Logs */}
                  <div className="rounded-lg overflow-hidden border">
                    <iframe
                      src={`${grafanaUrl.split('/d/')[0]}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=1&theme=dark`}
                      width="100%"
                      height="250"
                      frameBorder="0"
                      title="Application Logs"
                    />
                  </div>
                </>
              ) : (
                <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                  Loading Grafana panels...
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
