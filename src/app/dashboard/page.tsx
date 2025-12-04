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

import { Search, Maximize2, Minimize2, MessageSquare, Send, Calendar as CalendarIcon, Clock, ChevronLeft, ChevronRight } from "lucide-react"

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useMemo, useState } from "react";
import { DashboardRoleDialog } from "@/components/dashboard-role-dialog";
import { DashboardMessagingCard } from "@/components/dashboard-messaging-card";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"



export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [roleDialogOpen, setRoleDialogOpen] = useState(false);
  const [roleError, setRoleError] = useState<string | null>(null);
  const [isSavingRole, setIsSavingRole] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const [isExpanded, setIsExpanded] = useState(false);
  const [isCalendarExpanded, setIsCalendarExpanded] = useState(false);
  const [weekOffset, setWeekOffset] = useState(0);
  const [conversations, setConversations] = useState<Array<{
    id: string;
    name: string;
    avatar: string;
    avatarColor: string;
    lastMessage: string;
    time: string;
    unread: number;
    active: boolean;
  }>>([]);
  const [messages, setMessages] = useState<Array<{
    id: string;
    sender: string;
    message: string;
    time: string;
    isMe: boolean;
    avatar: string;
    avatarColor: string;
  }>>([]);


  // TODO: Replace with actual data from backend
  const mockAppointments: Array<{
    id: number;
    title: string;
    time: string;
    duration: string;
    client: string;
    color: string;
    startHour: number;
    durationMinutes: number;
    zoomLink?: string;
  }> = [
      { id: 1, title: "Therapy Session", time: "09:00 AM", duration: "1h", client: "Sarah Johnson", color: "bg-blue-500", startHour: 9, durationMinutes: 60, zoomLink: "https://zoom.us/j/1234567890" },
      { id: 2, title: "Initial Consultation", time: "11:00 AM", duration: "45m", client: "Michael Chen", color: "bg-emerald-500", startHour: 11, durationMinutes: 45, zoomLink: "https://zoom.us/j/0987654321" },
      { id: 3, title: "Follow-up", time: "02:00 PM", duration: "30m", client: "Emily Roberts", color: "bg-purple-500", startHour: 14, durationMinutes: 30, zoomLink: "https://zoom.us/j/1122334455" },
    ];

  const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const today = new Date();

  // Calculate the week start based on offset
  const getWeekStart = (offset: number) => {
    const date = new Date(today);
    const dayOfWeek = date.getDay();
    const diff = dayOfWeek === 0 ? -6 : 1 - dayOfWeek; // Adjust to Monday
    date.setDate(date.getDate() + diff + (offset * 7));
    return date;
  };

  const weekStart = getWeekStart(weekOffset);
  const currentMonth = weekStart.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  const currentDay = today.toLocaleDateString('en-US', { weekday: 'short' });

  // Generate week dates
  const weekDates = weekDays.map((_, index) => {
    const date = new Date(weekStart);
    date.setDate(weekStart.getDate() + index);
    return date;
  });

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  const shouldPromptForRole = useMemo(() => {
    if (!session?.user) return false;
    const role = (session.user as { role?: string | null }).role ?? "UNSET";
    return role === "UNSET" || !role;
  }, [session]);

  useEffect(() => {
    if (shouldPromptForRole) {
      setRoleDialogOpen(true);
    }
  }, [shouldPromptForRole]);

  async function handleRoleSelect(role: "practitioner" | "patient") {
    setRoleError(null);
    setIsSavingRole(true);
    try {
      const res = await fetch("/api/user/role", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setRoleError(body.error ?? "Unable to save your role. Please try again.");
      } else {
        setRoleDialogOpen(false);
        router.refresh();
      }
    } catch (err) {
      setRoleError("Unable to save your role. Please try again.");
    } finally {
      setIsSavingRole(false);
    }
  }

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;

    const loadMessages = async () => {
      try {
        const res = await fetch("/api/messages");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setConversations(data.conversations ?? []);
        setMessages(data.messages ?? []);
      } catch {
        // swallow errors to keep dashboard responsive
      }
    };

    loadMessages();
    return () => {
      cancelled = true;
    };
  }, [isPending, session]);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;
  return (

    <SidebarProvider>
      <AppSidebar user={user}
      />
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
        <div className="flex flex-1 flex-col gap-3 p-4 pt-4">
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <DashboardMessagingCard
              conversations={conversations}
              messages={messages}
            />
            {/* Calendar Preview Box */}
            <Card
              className={`transition-all duration-300 ${isCalendarExpanded
                ? 'fixed inset-0 z-50 rounded-none border-0 md:col-span-1'
                : 'md:col-span-1 aspect-video cursor-pointer hover:shadow-lg'
                }`}
              onClick={() => !isCalendarExpanded && setIsCalendarExpanded(true)}
            >
              {!isCalendarExpanded ? (
                // Collapsed Preview View
                <>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CalendarIcon className="h-5 w-5 text-primary" />
                        <CardTitle>Calendar</CardTitle>
                      </div>
                      <Maximize2 className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <CardDescription>
                      {today.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                    </CardDescription>
                    <p className="text-xs text-muted-foreground mt-1">
                      {mockAppointments.length} {mockAppointments.length === 1 ? 'appointment' : 'appointments'} today
                    </p>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-hidden pb-4">
                    <div className="space-y-1.5 h-full flex flex-col">
                      {mockAppointments.slice(0, 2).map((apt) => (
                        <div key={apt.id} className="flex items-center gap-3 p-1.5 rounded-lg hover:bg-muted/50 transition-colors">
                          <div className={`h-8 w-1 rounded-full ${apt.color}`}></div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold truncate">{apt.client}</p>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span>{apt.time} • {apt.duration}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                      {mockAppointments.length > 2 && (
                        <div className="flex pl-2 pt-0.5">
                          <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                            +{mockAppointments.length - 2} more
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </>
              ) : (
                // Expanded Weekly Calendar View
                <div className="flex flex-col h-screen">
                  <div className="p-4 border-b bg-background">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div>
                          <h2 className="text-2xl font-bold">Calendar</h2>
                          <p className="text-sm text-muted-foreground">{currentMonth}</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <Button
                            size="icon-sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              setWeekOffset(prev => prev - 1);
                            }}
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              setWeekOffset(0);
                            }}
                          >
                            Today
                          </Button>
                          <Button
                            size="icon-sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              setWeekOffset(prev => prev + 1);
                            }}
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          setIsCalendarExpanded(false);
                        }}
                      >
                        <Minimize2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Weekly Calendar Grid */}
                  <div className="flex-1 overflow-auto p-4">
                    <div className="grid grid-cols-8 gap-2 min-h-full">
                      {/* Time column */}
                      <div className="flex flex-col pt-14">
                        {Array.from({ length: 15 }, (_, i) => i + 8).map((hour) => (
                          <div key={hour} className="text-xs text-muted-foreground text-right pr-2 h-16 flex items-start">
                            {hour > 12 ? `${hour - 12}:00 PM` : hour === 12 ? '12:00 PM' : `${hour}:00 AM`}
                          </div>
                        ))}
                      </div>

                      {/* Week days columns */}
                      {weekDays.map((day, dayIndex) => {
                        const date = weekDates[dayIndex];
                        const dateNumber = date.getDate();
                        const isToday = date.toDateString() === today.toDateString();

                        return (
                          <div key={day} className="flex flex-col">
                            <div className={`text-center pb-2 mb-2 border-b h-14 flex flex-col justify-center ${isToday ? 'font-bold text-primary' : 'text-muted-foreground'}`}>
                              <div className="text-xs mb-1">{day}</div>
                              <div className={`text-lg ${isToday ? 'bg-primary text-primary-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto' : ''}`}>
                                {dateNumber}
                              </div>
                            </div>

                            {/* Time slots */}
                            <div className="flex-1 relative">
                              <div className="absolute inset-0 flex flex-col">
                                {Array.from({ length: 15 }).map((_, i) => (
                                  <div key={i} className="h-16 border-t border-muted/30"></div>
                                ))}
                              </div>

                              {/* Appointments - only show on current day for demo */}
                              {isToday && mockAppointments.map((apt, idx) => {
                                const startOffset = ((apt.startHour - 8) * 60) / (15 * 60) * 100; // Calculate percentage from 8 AM start (15 hours total)
                                const height = (apt.durationMinutes / (15 * 60)) * 100; // Calculate height as percentage of 15-hour range

                                // Calculate end time
                                const endHour = apt.startHour + Math.floor(apt.durationMinutes / 60);
                                const endMinutes = apt.durationMinutes % 60;
                                const endTime = `${endHour > 12 ? endHour - 12 : endHour}:${endMinutes.toString().padStart(2, '0')} ${endHour >= 12 ? 'PM' : 'AM'}`;

                                return (
                                  <div
                                    key={apt.id}
                                    className={`absolute ${apt.color} text-white rounded-lg p-2 left-0 right-0 mx-1 cursor-pointer hover:shadow-lg transition-shadow`}
                                    style={{
                                      top: `${startOffset}%`,
                                      height: `${height}%`,
                                      minHeight: '50px'
                                    }}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (apt.zoomLink) {
                                        window.open(apt.zoomLink, '_blank', 'noopener,noreferrer');
                                      }
                                    }}
                                  >
                                    <p className="text-xs font-semibold line-clamp-1">{apt.client}</p>
                                    <p className="text-xs opacity-90">{apt.time} - {endTime}</p>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </Card>
            <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
          </div>
          <Skeleton className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" />
        </div>
      </SidebarInset>
      <DashboardRoleDialog
        open={roleDialogOpen}
        error={roleError}
        isSaving={isSavingRole}
        onSelect={handleRoleSelect}
      />
    </SidebarProvider>
  )
}
