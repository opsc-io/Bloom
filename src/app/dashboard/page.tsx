"use client";

import { AppSidebar } from "@/components/app-sidebar"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";
import { Users, UserCheck, ShieldCheck, Activity, TrendingUp, Clock } from "lucide-react"
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
} from "recharts";

interface AdminStats {
  overview: {
    totalUsers: number;
    therapistCount: number;
    adminCount: number;
    patientCount: number;
    newUsersThisWeek: number;
    activeSessions: number;
  };
  recentUsers: {
    id: string;
    firstname: string;
    lastname: string;
    email: string;
    createdAt: string;
    therapist: boolean | null;
    administrator: boolean | null;
  }[];
  authMethods: { method: string; count: number }[];
  userGrowth: { date: string; users: number }[];
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"];

export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  // Fetch admin stats when user is admin
  useEffect(() => {
    const fetchStats = async () => {
      if (session?.user?.administrator) {
        try {
          setStatsLoading(true);
          const response = await fetch("/api/admin/stats");
          if (!response.ok) {
            throw new Error("Failed to fetch stats");
          }
          const data = await response.json();
          setStats(data);
          setStatsError(null);
        } catch (err) {
          setStatsError(err instanceof Error ? err.message : "Failed to load stats");
        } finally {
          setStatsLoading(false);
        }
      }
    };

    if (session?.user?.administrator) {
      fetchStats();
      // Refresh every 30 seconds
      const interval = setInterval(fetchStats, 30000);
      return () => clearInterval(interval);
    }
  }, [session?.user?.administrator]);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;
  const isAdmin = user.administrator === true;

  // Process user role distribution for pie chart
  const roleData = stats ? [
    { name: "Patients", value: stats.overview.patientCount },
    { name: "Therapists", value: stats.overview.therapistCount },
    { name: "Admins", value: stats.overview.adminCount },
  ].filter(d => d.value > 0) : [];

  return (
    <SidebarProvider>
      <AppSidebar user={user} />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator
              orientation="vertical"
              className="mr-2 data-[orientation=vertical]:h-4"
            />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">
                    Dashboard
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          {isAdmin ? (
            <>
              {/* Admin Dashboard - Stats Cards */}
              <div className="grid auto-rows-min gap-4 md:grid-cols-3">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                    <Users className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    {statsLoading ? (
                      <Skeleton className="h-8 w-20" />
                    ) : (
                      <>
                        <div className="text-2xl font-bold">{stats?.overview.totalUsers ?? 0}</div>
                        <p className="text-xs text-muted-foreground">
                          +{stats?.overview.newUsersThisWeek ?? 0} this week
                        </p>
                      </>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Therapists</CardTitle>
                    <UserCheck className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    {statsLoading ? (
                      <Skeleton className="h-8 w-20" />
                    ) : (
                      <>
                        <div className="text-2xl font-bold">{stats?.overview.therapistCount ?? 0}</div>
                        <p className="text-xs text-muted-foreground">Active practitioners</p>
                      </>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
                    <Activity className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    {statsLoading ? (
                      <Skeleton className="h-8 w-20" />
                    ) : (
                      <>
                        <div className="text-2xl font-bold">{stats?.overview.activeSessions ?? 0}</div>
                        <p className="text-xs text-muted-foreground">Currently logged in</p>
                      </>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Charts Row */}
              <div className="grid gap-4 md:grid-cols-2">
                {/* User Growth Chart */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-base">User Growth</CardTitle>
                        <CardDescription>New signups over the last 30 days</CardDescription>
                      </div>
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    {statsLoading ? (
                      <Skeleton className="h-[200px] w-full" />
                    ) : statsError ? (
                      <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                        {statsError}
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={stats?.userGrowth ?? []}>
                          <defs>
                            <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis
                            dataKey="date"
                            tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                            tick={{ fontSize: 12 }}
                            className="text-muted-foreground"
                          />
                          <YAxis tick={{ fontSize: 12 }} className="text-muted-foreground" />
                          <Tooltip
                            labelFormatter={(value) => new Date(value).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}
                            contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))" }}
                          />
                          <Area
                            type="monotone"
                            dataKey="users"
                            stroke="#8884d8"
                            fillOpacity={1}
                            fill="url(#colorUsers)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}
                  </CardContent>
                </Card>

                {/* User Distribution Chart */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-base">User Distribution</CardTitle>
                        <CardDescription>Breakdown by role</CardDescription>
                      </div>
                      <ShieldCheck className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    {statsLoading ? (
                      <Skeleton className="h-[200px] w-full" />
                    ) : statsError ? (
                      <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                        {statsError}
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                          <Pie
                            data={roleData}
                            cx="50%"
                            cy="50%"
                            innerRadius={40}
                            outerRadius={80}
                            fill="#8884d8"
                            paddingAngle={5}
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                          >
                            {roleData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))" }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Recent Users Table */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Recent Signups</CardTitle>
                      <CardDescription>Latest users to join the platform</CardDescription>
                    </div>
                    <Clock className="h-4 w-4 text-muted-foreground" />
                  </div>
                </CardHeader>
                <CardContent>
                  {statsLoading ? (
                    <div className="space-y-2">
                      {[...Array(5)].map((_, i) => (
                        <Skeleton key={i} className="h-12 w-full" />
                      ))}
                    </div>
                  ) : statsError ? (
                    <div className="text-center py-4 text-muted-foreground">{statsError}</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Name</th>
                            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Email</th>
                            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Role</th>
                            <th className="text-left py-2 px-2 font-medium text-muted-foreground">Joined</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats?.recentUsers.map((user) => (
                            <tr key={user.id} className="border-b last:border-0">
                              <td className="py-2 px-2">{user.firstname} {user.lastname}</td>
                              <td className="py-2 px-2 text-muted-foreground">{user.email}</td>
                              <td className="py-2 px-2">
                                <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${
                                  user.administrator
                                    ? "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300"
                                    : user.therapist
                                    ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                                    : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                                }`}>
                                  {user.administrator ? "Admin" : user.therapist ? "Therapist" : "Patient"}
                                </span>
                              </td>
                              <td className="py-2 px-2 text-muted-foreground">
                                {new Date(user.createdAt).toLocaleDateString("en-US", {
                                  month: "short",
                                  day: "numeric",
                                  year: "numeric",
                                })}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          ) : (
            <>
              {/* Regular User Dashboard - Placeholder */}
              <div className="grid auto-rows-min gap-4 md:grid-cols-3">
                <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
                <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
                <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
              </div>
              <Skeleton className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" />
            </>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
