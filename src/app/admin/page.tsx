"use client";

import { AppSidebar } from "@/components/app-sidebar"
import { NavUser } from "@/components/nav-user"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState, Suspense } from "react";
import Image from "next/image"
import {
  Users,
  Stethoscope,
  Sparkles,
  TrendingUp,
  Clock,
  ExternalLink,
  BarChart3,
  Database,
  Container,
  HardDrive,
  FileText,
  LayoutDashboard
} from "lucide-react"
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
import { cn } from "@/lib/utils";

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

const COLORS = ["#60a5fa", "#4CBB17", "#f472b6"];

type TabType = "overview" | "database" | "containers" | "redis" | "logs";

const tabs: { id: TabType; label: string; icon: React.ElementType }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "database", label: "Database", icon: Database },
  { id: "containers", label: "Containers", icon: Container },
  { id: "redis", label: "Redis Cache", icon: HardDrive },
  { id: "logs", label: "System Logs", icon: FileText },
];

function AdminContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const [grafanaUrl, setGrafanaUrl] = useState('');
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  const activeTab = (searchParams.get("tab") as TabType) || "overview";

  const setActiveTab = (tab: TabType) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", tab);
    router.push(`/admin?${params.toString()}`);
  };

  useEffect(() => {
    const hostname = window.location.hostname;
    const isProduction = hostname === 'bloomhealth.us' || hostname === 'www.bloomhealth.us';
    const isQa = hostname === 'qa.bloomhealth.us' || hostname === 'qa.gcp.bloomhealth.us';
    const isDev = hostname === 'dev.gcp.bloomhealth.us';

    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      setGrafanaUrl('http://localhost:3001/d/bloom-overview/bloom-overview');
    } else if (isProduction) {
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-production/bloom-production');
    } else if (isQa) {
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa');
    } else if (isDev) {
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-dev/bloom-dev');
    } else {
      // Default to Grafana Cloud for unknown GKE hostnames
      setGrafanaUrl('https://opscvisuals.grafana.net/d/bloom-overview/bloom-overview');
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
    { name: "Therapists", value: stats.overview.therapistCount },
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

  const grafanaBase = grafanaUrl?.split('/d/')[0] || '';

  return (
    <SidebarProvider>
      {/* Custom Admin Sidebar */}
      <Sidebar>
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton size="lg" asChild>
                <a href="/dashboard">
                  <div className="flex items-center justify-center">
                    <Image src="/logo.svg" alt="Bloom Logo" width={100} height={33} className="h-6 w-auto" />
                  </div>
                  <div className="flex flex-col gap-0.5 leading-none">
                    <span className="font-semibold">Admin Panel</span>
                    <span className="text-xs text-muted-foreground">System Management</span>
                  </div>
                </a>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        <SidebarContent>
          <nav className="space-y-1 p-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    activeTab === tab.id
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
          {/* Quick Links */}
          <div className="mt-auto p-4 border-t">
            <p className="text-xs font-medium text-muted-foreground mb-3">Quick Links</p>
            <Button variant="ghost" size="sm" className="w-full justify-start" asChild>
              <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                <BarChart3 className="mr-2 h-4 w-4" />
                Grafana
                <ExternalLink className="ml-auto h-3 w-3" />
              </a>
            </Button>
          </div>
        </SidebarContent>
        <SidebarFooter>
          <NavUser user={user} />
        </SidebarFooter>
      </Sidebar>
      
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbLink href="/admin">Admin</BreadcrumbLink>
                </BreadcrumbItem>
                {activeTab !== "overview" && (
                  <>
                    <BreadcrumbSeparator />
                    <BreadcrumbItem>
                      <span className="capitalize">{activeTab}</span>
                    </BreadcrumbItem>
                  </>
                )}
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto p-4">
            {/* Overview Tab */}
            {activeTab === "overview" && (
              <div className="space-y-4">
                <h2 className="text-2xl font-bold">Admin Overview</h2>

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
                    <CardContent className="h-[250px]">
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
                              cy="45%"
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
                            <Legend verticalAlign="bottom" height={36} />
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
              </div>
            )}

            {/* Database Tab */}
            {activeTab === "database" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Database Performance</h2>
                    <p className="text-muted-foreground">CockroachDB metrics and health</p>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`${grafanaBase}/d/bloom-cockroachdb/cockroachdb`} target="_blank" rel="noopener noreferrer">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Full Dashboard <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                </div>

                {grafanaBase ? (
                  <>
                    {/* Database Stats Row */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=1&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Active Connections"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=2&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Live Nodes"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=3&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Storage Used %"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=4&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Storage Used"
                        />
                      </div>
                    </div>

                    {/* SQL Operations & Latency */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=5&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="SQL Operations Rate"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=6&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Query Latency"
                          />
                        </div>
                      </Card>
                    </div>

                    {/* Transactions */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=7&theme=light`}
                            width="100%"
                            height="250"
                            frameBorder="0"
                            title="Transactions"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-cockroachdb/cockroachdb?orgId=1&panelId=8&theme=light`}
                            width="100%"
                            height="250"
                            frameBorder="0"
                            title="Connections & Open Transactions"
                          />
                        </div>
                      </Card>
                    </div>
                  </>
                ) : (
                  <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                    Loading database metrics...
                  </div>
                )}
              </div>
            )}

            {/* Containers Tab */}
            {activeTab === "containers" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Container Resources</h2>
                    <p className="text-muted-foreground">CPU, memory, network, and disk I/O metrics</p>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`${grafanaBase}/d/bloom-containers/container-resources`} target="_blank" rel="noopener noreferrer">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Full Dashboard <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                </div>

                {grafanaBase ? (
                  <>
                    {/* CPU & Memory */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-containers/container-resources?orgId=1&panelId=1&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="CPU Usage by Container"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-containers/container-resources?orgId=1&panelId=2&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Memory Usage by Container"
                          />
                        </div>
                      </Card>
                    </div>

                    {/* Network & Disk I/O */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-containers/container-resources?orgId=1&panelId=3&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Network I/O"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-containers/container-resources?orgId=1&panelId=4&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Disk I/O"
                          />
                        </div>
                      </Card>
                    </div>

                    {/* Container Restarts */}
                    <Card className="p-0 overflow-hidden">
                      <div className="bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-containers/container-resources?orgId=1&panelId=5&theme=light`}
                          width="100%"
                          height="250"
                          frameBorder="0"
                          title="Container Restarts (24h)"
                        />
                      </div>
                    </Card>
                  </>
                ) : (
                  <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                    Loading container metrics...
                  </div>
                )}
              </div>
            )}

            {/* Redis Tab */}
            {activeTab === "redis" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Redis Cache</h2>
                    <p className="text-muted-foreground">Cache performance, memory usage, and pub/sub activity</p>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`${grafanaBase}/d/bloom-redis/redis-metrics`} target="_blank" rel="noopener noreferrer">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Full Dashboard <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                </div>

                {grafanaBase ? (
                  <>
                    {/* Redis Stats Row */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=1&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Redis Status"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=2&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Memory Used"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=3&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Connected Clients"
                        />
                      </div>
                      <div className="rounded-lg overflow-hidden border bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=4&theme=light`}
                          width="100%"
                          height="120"
                          frameBorder="0"
                          title="Total Keys (db0)"
                        />
                      </div>
                    </div>

                    {/* Commands & Hit Rate */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=5&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Commands per Second"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=6&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Cache Hit Rate"
                          />
                        </div>
                      </Card>
                    </div>

                    {/* Memory & Pub/Sub */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=7&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Memory Usage"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-redis/redis-metrics?orgId=1&panelId=8&theme=light`}
                            width="100%"
                            height="300"
                            frameBorder="0"
                            title="Pub/Sub Activity"
                          />
                        </div>
                      </Card>
                    </div>
                  </>
                ) : (
                  <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                    Loading Redis metrics...
                  </div>
                )}
              </div>
            )}

            {/* Logs Tab */}
            {activeTab === "logs" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">System Logs</h2>
                    <p className="text-muted-foreground">Live logs, errors, and warnings from all containers</p>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`${grafanaBase}/d/bloom-overview/bloom-overview`} target="_blank" rel="noopener noreferrer">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Full Dashboard <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                </div>

                {grafanaBase ? (
                  <>
                    {/* Log Volume & Errors */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=4&theme=light`}
                            width="100%"
                            height="250"
                            frameBorder="0"
                            title="Log Volume by Container"
                          />
                        </div>
                      </Card>
                      <Card className="p-0 overflow-hidden">
                        <div className="bg-white">
                          <iframe
                            src={`${grafanaBase}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=5&theme=light`}
                            width="100%"
                            height="250"
                            frameBorder="0"
                            title="Errors and Warnings"
                          />
                        </div>
                      </Card>
                    </div>

                    {/* Application Logs */}
                    <Card className="p-0 overflow-hidden">
                      <div className="bg-white">
                        <iframe
                          src={`${grafanaBase}/d-solo/bloom-overview/bloom-overview?orgId=1&panelId=1&theme=light`}
                          width="100%"
                          height="500"
                          frameBorder="0"
                          title="Application Logs"
                        />
                      </div>
                    </Card>
                  </>
                ) : (
                  <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                    Loading logs...
                  </div>
                )}
              </div>
            )}
          </main>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default function AdminPage() {
  return (
    <Suspense fallback={<p className="text-center mt-8">Loading...</p>}>
      <AdminContent />
    </Suspense>
  );
}
