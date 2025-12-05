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
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";
import { Users, Activity, ExternalLink, TrendingUp, Clock, ChartPie } from "lucide-react"

interface AdminStats {
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
    firstname: string;
    lastname: string;
    email: string;
    createdAt: string;
    therapist: boolean;
    administrator: boolean;
  }>;
  authMethods: Array<{
    method: string;
    count: number;
  }>;
  userGrowth: Array<{
    date: string;
    users: number;
  }>;
}

// Grafana dashboard URLs based on environment
const getGrafanaUrl = () => {
  const isProduction = typeof window !== 'undefined' &&
    (window.location.hostname === 'bloomhealth.us' || window.location.hostname === 'www.bloomhealth.us');

  return isProduction
    ? 'https://opscvisuals.grafana.net/d/bloom-production/bloom-production'
    : 'https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa';
};

export default function AdminPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  // Check if user is administrator
  const isAdmin = session?.user?.administrator === true;

  // Fetch admin stats when user is authenticated and is admin
  useEffect(() => {
    if (isAdmin) {
      fetch('/api/admin/stats')
        .then(res => res.json())
        .then(data => {
          if (!data.error) {
            setStats(data);
          }
        })
        .catch(console.error)
        .finally(() => setStatsLoading(false));
    }
  }, [isAdmin]);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;

  // If not admin, show access denied
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

  const grafanaUrl = getGrafanaUrl();

  // Calculate chart dimensions
  const maxUsers = Math.max(...(stats?.userGrowth?.map(d => d.users) || [1]), 1);
  const chartHeight = 120;
  const chartWidth = 100; // percentage

  // Get role for display
  const getRoleDisplay = (user: AdminStats['recentUsers'][0]) => {
    if (user.administrator) return { text: 'Admin', color: 'bg-purple-100 text-purple-700' };
    if (user.therapist) return { text: 'Therapist', color: 'bg-blue-100 text-blue-700' };
    return { text: 'Patient', color: 'bg-gray-100 text-gray-700' };
  };

  // Calculate percentages for pie chart
  const total = stats?.overview.totalUsers || 1;
  const therapistPct = Math.round(((stats?.overview.therapistCount || 0) / total) * 100);
  const adminPct = Math.round(((stats?.overview.adminCount || 0) / total) * 100);
  const patientPct = 100 - therapistPct - adminPct;

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
          {/* Top Stats Row */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {statsLoading ? "..." : stats?.overview.totalUsers ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  +{stats?.overview.newUsersThisWeek ?? 0} this week
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Therapists</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {statsLoading ? "..." : stats?.overview.therapistCount ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">Active practitioners</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {statsLoading ? "..." : stats?.overview.activeSessions ?? 0}
                </div>
                <p className="text-xs text-muted-foreground">Currently logged in</p>
              </CardContent>
            </Card>
          </div>

          {/* Charts Row */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* User Growth Chart */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-medium">User Growth</CardTitle>
                  <CardDescription>New signups over the last 30 days</CardDescription>
                </div>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {statsLoading ? (
                  <div className="h-[140px] flex items-center justify-center text-muted-foreground">Loading...</div>
                ) : (
                  <div className="h-[140px] flex items-end gap-[2px]">
                    {stats?.userGrowth?.map((day, i) => (
                      <div
                        key={day.date}
                        className="flex-1 rounded-t transition-colors cursor-pointer"
                        style={{
                          height: `${Math.max((day.users / maxUsers) * chartHeight, day.users > 0 ? 8 : 2)}px`,
                          minHeight: day.users > 0 ? '8px' : '2px',
                          backgroundColor: day.users > 0 ? '#14b8a6' : '#e5e7eb'
                        }}
                        title={`${day.date}: ${day.users} users`}
                      />
                    ))}
                  </div>
                )}
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>{stats?.userGrowth?.[0]?.date ? new Date(stats.userGrowth[0].date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''}</span>
                  <span>{stats?.userGrowth?.[stats.userGrowth.length - 1]?.date ? new Date(stats.userGrowth[stats.userGrowth.length - 1].date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : ''}</span>
                </div>
              </CardContent>
            </Card>

            {/* User Distribution */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-medium">User Distribution</CardTitle>
                  <CardDescription>Breakdown by role</CardDescription>
                </div>
                <ChartPie className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {statsLoading ? (
                  <div className="h-[140px] flex items-center justify-center text-muted-foreground">Loading...</div>
                ) : (
                  <div className="flex items-center justify-center gap-8">
                    {/* Simple Donut Chart */}
                    <div className="relative w-32 h-32">
                      <svg viewBox="0 0 100 100" className="transform -rotate-90">
                        {/* Background circle */}
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#e5e7eb" strokeWidth="20" />
                        {/* Patient segment - Teal */}
                        <circle
                          cx="50" cy="50" r="40" fill="none"
                          stroke="#14b8a6"
                          strokeWidth="20"
                          strokeDasharray={`${patientPct * 2.51} 251`}
                          strokeDashoffset="0"
                        />
                        {/* Admin segment - Cyan */}
                        <circle
                          cx="50" cy="50" r="40" fill="none"
                          stroke="#06b6d4"
                          strokeWidth="20"
                          strokeDasharray={`${adminPct * 2.51} 251`}
                          strokeDashoffset={`${-patientPct * 2.51}`}
                        />
                        {/* Therapist segment - Blue */}
                        {therapistPct > 0 && (
                          <circle
                            cx="50" cy="50" r="40" fill="none"
                            stroke="#3b82f6"
                            strokeWidth="20"
                            strokeDasharray={`${therapistPct * 2.51} 251`}
                            strokeDashoffset={`${-(patientPct + adminPct) * 2.51}`}
                          />
                        )}
                      </svg>
                    </div>
                    {/* Legend */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#14b8a6' }} />
                        <span className="text-sm text-teal-600">Patients {patientPct}%</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#06b6d4' }} />
                        <span className="text-sm text-cyan-600">Admins {adminPct}%</span>
                      </div>
                      {therapistPct > 0 && (
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#3b82f6' }} />
                          <span className="text-sm text-blue-600">Therapists {therapistPct}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recent Signups Table */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-sm font-medium">Recent Signups</CardTitle>
                <CardDescription>Latest users to join the platform</CardDescription>
              </div>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <p className="text-muted-foreground">Loading...</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Name</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Email</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Role</th>
                        <th className="text-left py-3 px-2 text-sm font-medium text-muted-foreground">Joined</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats?.recentUsers.map((user) => {
                        const role = getRoleDisplay(user);
                        return (
                          <tr key={user.id} className="border-b last:border-0">
                            <td className="py-3 px-2 font-medium">{user.firstname} {user.lastname}</td>
                            <td className="py-3 px-2 text-muted-foreground">{user.email}</td>
                            <td className="py-3 px-2">
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${role.color}`}>
                                {role.text}
                              </span>
                            </td>
                            <td className="py-3 px-2 text-muted-foreground">
                              {new Date(user.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Grafana Analytics Card */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-sm font-medium">Grafana Analytics</CardTitle>
                <CardDescription>Live dashboard panels from Grafana</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                  <ChartPie className="mr-2 h-4 w-4" />
                  Open Grafana <ExternalLink className="ml-1 h-3 w-3" />
                </a>
              </Button>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <ChartPie className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">Grafana panels will appear here once configured</p>
                <Button variant="outline" asChild>
                  <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                    Configure in Grafana <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
