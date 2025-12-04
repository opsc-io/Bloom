"use client";

import { AppSidebar } from "@/components/app-sidebar"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";
import {
  Users, UserCheck, ShieldCheck, Activity, TrendingUp, Clock, BarChart3, ExternalLink,
  Search, Maximize2, Minimize2, MessageSquare, Send
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
  const [grafanaPanels, setGrafanaPanels] = useState<{ id: number; title: string; type: string }[]>([]);
  const [grafanaLoading, setGrafanaLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);

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

  // Fetch Grafana panels
  useEffect(() => {
    const fetchGrafanaPanels = async () => {
      if (session?.user?.administrator) {
        try {
          setGrafanaLoading(true);
          const response = await fetch("/api/admin/grafana?endpoint=panels");
          if (response.ok) {
            const data = await response.json();
            setGrafanaPanels(data.panels || []);
          }
        } catch (err) {
          console.error("Failed to fetch Grafana panels:", err);
        } finally {
          setGrafanaLoading(false);
        }
      }
    };

    if (session?.user?.administrator) {
      fetchGrafanaPanels();
    }
  }, [session?.user?.administrator]);

  // TODO: Replace with actual data from backend
  const mockMessages: Array<{
    id: number;
    sender: string;
    message: string;
    time: string;
    isMe: boolean;
    avatar: string;
    avatarColor: string;
  }> = [
    { id: 1, sender: "Dr. Sarah Johnson", message: "Hi, I have a question about the upcoming appointment.", time: "10:30 AM", isMe: false, avatar: "SJ", avatarColor: "bg-purple-500" },
    { id: 2, sender: "You", message: "Of course! How can I help you?", time: "10:32 AM", isMe: true, avatar: session?.user?.name?.[0] || "U", avatarColor: "bg-blue-500" },
    { id: 3, sender: "Dr. Sarah Johnson", message: "Can we reschedule to next Tuesday?", time: "10:33 AM", isMe: false, avatar: "SJ", avatarColor: "bg-purple-500" },
  ];

  // TODO: Replace with actual data from backend
  const mockConversations: Array<{
    id: number;
    name: string;
    avatar: string;
    avatarColor: string;
    lastMessage: string;
    time: string;
    unread: number;
    active: boolean;
  }> = [
    { id: 1, name: "Dr. Sarah Johnson", avatar: "SJ", avatarColor: "bg-purple-500", lastMessage: "Can we reschedule to next Tuesday?", time: "10:33 AM", unread: 1, active: true },
  ];

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
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4 justify-between">
          <div className="flex items-center gap-2">
            <SidebarTrigger className="-ml-1" />
            <Separator
              orientation="vertical"
              className="mr-2 data-[orientation=vertical]:h-4"
            />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">
                    Building Your Application
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage>Data Fetching</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative w-64">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search therapists..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          {isAdmin ? (
            <div>
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
                      <div>
                        <div className="text-2xl font-bold">{stats?.overview.totalUsers ?? 0}</div>
                        <p className="text-xs text-muted-foreground">
                          +{stats?.overview.newUsersThisWeek ?? 0} this week
                        </p>
                      </div>
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
                      <div>
                        <div className="text-2xl font-bold">{stats?.overview.therapistCount ?? 0}</div>
                        <p className="text-xs text-muted-foreground">Active practitioners</p>
                      </div>
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
                      <div>
                        <div className="text-2xl font-bold">{stats?.overview.activeSessions ?? 0}</div>
                        <p className="text-xs text-muted-foreground">Currently logged in</p>
                      </div>
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

              {/* Grafana Panels Section */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Grafana Analytics</CardTitle>
                      <CardDescription>Live dashboard panels from Grafana</CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-muted-foreground" />
                      <Button variant="outline" size="sm" asChild>
                        <a
                          href="https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Open Grafana <ExternalLink className="ml-1 h-3 w-3" />
                        </a>
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {grafanaLoading ? (
                    <div className="grid gap-4 md:grid-cols-2">
                      <Skeleton className="h-[300px] w-full" />
                      <Skeleton className="h-[300px] w-full" />
                    </div>
                  ) : grafanaPanels.length > 0 ? (
                    <div className="grid gap-4 md:grid-cols-2">
                      {grafanaPanels.slice(0, 4).map((panel) => (
                        <div key={panel.id} className="relative">
                          <div className="text-sm font-medium mb-2">{panel.title}</div>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`/api/admin/grafana?endpoint=render&panelId=${panel.id}&width=600&height=300`}
                            alt={panel.title}
                            className="w-full h-auto rounded-lg border bg-background"
                            loading="lazy"
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Grafana panels will appear here once configured</p>
                      <Button variant="outline" className="mt-4" asChild>
                        <a
                          href="https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Configure in Grafana <ExternalLink className="ml-1 h-3 w-3" />
                        </a>
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          ) : (
            <div>
              {/* Regular User Dashboard - Messages UI */}
              <div className="grid auto-rows-min gap-4 md:grid-cols-3">
                {/* Chat Messages Preview Box */}
                <Card
                  className={`transition-all duration-300 ${
                    isExpanded
                      ? 'fixed inset-0 z-50 rounded-none border-0 md:col-span-1'
                      : 'md:col-span-1 aspect-video cursor-pointer hover:shadow-lg'
                  }`}
                  onClick={() => !isExpanded && setIsExpanded(true)}
                >
                  {!isExpanded ? (
                    // Collapsed Preview View
                    <div>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <MessageSquare className="h-5 w-5 text-primary" />
                            <CardTitle>Messages</CardTitle>
                          </div>
                          <Maximize2 className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <CardDescription>Recent conversations</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center gap-3 flex-wrap">
                          {mockConversations.length > 0 ? (
                            mockConversations.map((conv) => (
                              <div key={conv.id} className="relative">
                                <Avatar className="h-12 w-12 cursor-pointer hover:scale-110 transition-transform">
                                  <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                                    {conv.avatar}
                                  </AvatarFallback>
                                </Avatar>
                                {conv.unread > 0 && (
                                  <span className="absolute -top-1 -right-1 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center font-semibold border-2 border-background">
                                    {conv.unread}
                                  </span>
                                )}
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-muted-foreground py-4">No messages yet</p>
                          )}
                        </div>
                      </CardContent>
                    </div>
                  ) : (
                    // Expanded Full-Screen Chat View
                    <div className="flex h-screen">
                      {/* Left Sidebar - Conversations List */}
                      <div className="w-80 border-r bg-muted/30 flex flex-col">
                        <div className="p-4 border-b">
                          <div className="flex items-center justify-between mb-4">
                            <h2 className="text-2xl font-bold">Chats</h2>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                setIsExpanded(false);
                              }}
                            >
                              <Minimize2 className="h-4 w-4" />
                            </Button>
                          </div>
                          <div className="relative">
                            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                            <Input placeholder="Search conversations..." className="pl-8" />
                          </div>
                        </div>
                        <div className="flex-1 overflow-y-auto">
                          {mockConversations.length > 0 ? (
                            mockConversations.map((conv) => (
                              <div
                                key={conv.id}
                                className={`flex items-center gap-3 p-4 hover:bg-muted/50 cursor-pointer transition-colors ${
                                  conv.active ? 'bg-muted/50 border-l-2 border-primary' : ''
                                }`}
                              >
                                <Avatar className="h-12 w-12 shrink-0">
                                  <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                                    {conv.avatar}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between mb-1">
                                    <p className="font-semibold text-sm truncate">{conv.name}</p>
                                    <span className="text-xs text-muted-foreground">{conv.time}</span>
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <p className="text-sm text-muted-foreground truncate">{conv.lastMessage}</p>
                                    {conv.unread > 0 && (
                                      <span className="ml-2 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center shrink-0">
                                        {conv.unread}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-muted-foreground">No conversations yet</p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right Side - Active Conversation */}
                      <div className="flex-1 flex flex-col">
                        {/* Chat Header */}
                        <div className="p-4 border-b bg-background">
                          {mockConversations.find(c => c.active) ? (
                            <div className="flex items-center gap-3">
                              <Avatar className="h-10 w-10">
                                <AvatarFallback className={`${mockConversations.find(c => c.active)?.avatarColor} text-white font-semibold`}>
                                  {mockConversations.find(c => c.active)?.avatar}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <p className="font-semibold">{mockConversations.find(c => c.active)?.name}</p>
                                <p className="text-xs text-muted-foreground">Active now</p>
                              </div>
                            </div>
                          ) : (
                            <p className="text-muted-foreground">Select a conversation</p>
                          )}
                        </div>

                        {/* Messages Area */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-muted/20">
                          {mockMessages.length > 0 ? (
                            mockMessages.map((msg) => (
                              <div
                                key={msg.id}
                                className={`flex gap-3 ${msg.isMe ? 'justify-end' : 'justify-start'}`}
                              >
                                {!msg.isMe && (
                                  <Avatar className="h-8 w-8 shrink-0">
                                    <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                                      {msg.avatar}
                                    </AvatarFallback>
                                  </Avatar>
                                )}
                                <div className={`flex flex-col ${msg.isMe ? 'items-end' : 'items-start'} max-w-[60%]`}>
                                  {!msg.isMe && <p className="text-xs font-semibold text-muted-foreground mb-1">{msg.sender}</p>}
                                  <div className={`${msg.isMe ? 'bg-primary text-primary-foreground' : 'bg-background border'} rounded-2xl px-4 py-2.5 shadow-sm`}>
                                    <p className="text-sm">{msg.message}</p>
                                  </div>
                                  <p className="text-xs text-muted-foreground mt-1">{msg.time}</p>
                                </div>
                                {msg.isMe && (
                                  <Avatar className="h-8 w-8 shrink-0">
                                    <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                                      {msg.avatar}
                                    </AvatarFallback>
                                  </Avatar>
                                )}
                              </div>
                            ))
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <p className="text-muted-foreground">No messages yet</p>
                            </div>
                          )}
                        </div>

                        {/* Message Input Area */}
                        <div className="p-4 border-t bg-background">
                          <div className="flex gap-2">
                            <Input placeholder="Type a message..." className="flex-1" />
                            <Button size="icon" className="shrink-0">
                              <Send className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </Card>
                <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
                <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
              </div>
              <Skeleton className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" />
            </div>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
