"use client";

import { AppSidebar } from "@/components/app-sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Clock } from "lucide-react";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState, useMemo } from "react";

type Appointment = {
  id: string;
  title: string;
  start: string | Date;
  end: string | Date;
  durationMinutes: number;
  client: string;
  color: string;
  zoomLink?: string;
};

export default function CalendarPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [weekOffset, setWeekOffset] = useState(0);

  const today = useMemo(() => new Date(), []);
  const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  const getWeekStart = (offset: number) => {
    const date = new Date(today);
    const dayOfWeek = date.getDay();
    const diff = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
    date.setDate(date.getDate() + diff + offset * 7);
    date.setHours(0, 0, 0, 0);
    return date;
  };

  const weekStart = getWeekStart(weekOffset);
  const weekDates = weekDays.map((_, index) => {
    const date = new Date(weekStart);
    date.setDate(weekStart.getDate() + index);
    return date;
  });

  const currentMonth = weekStart.toLocaleDateString("en-US", { month: "long", year: "numeric" });

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;

    const loadAppointments = async () => {
      try {
        const res = await fetch("/api/appointments");
        if (!res.ok) return;
        const data = await res.json() as { appointments?: Appointment[] };
        if (cancelled) return;
        setAppointments(
          (data.appointments ?? []).map((apt) => ({
            ...apt,
            start: new Date(apt.start),
            end: new Date(apt.end),
          }))
        );
      } catch {
        // keep calendar usable even if appointments fail
      }
    };

    loadAppointments();
    return () => {
      cancelled = true;
    };
  }, [isPending, session]);

  const appointmentsThisWeek = useMemo(() => {
    return appointments
      .filter((apt) => {
        const start = apt.start as Date;
        return start >= weekStart && start < new Date(weekStart.getTime() + 7 * 24 * 60 * 60 * 1000);
      })
      .sort((a, b) => (a.start as Date).getTime() - (b.start as Date).getTime());
  }, [appointments, weekStart]);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;

  return (
    <SidebarProvider>
      <AppSidebar user={user} />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem className="hidden md:block">
                <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem>
                <BreadcrumbPage>Calendar</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden">
          <div className="p-4 border-b bg-background flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <CalendarIcon className="h-6 w-6 text-primary" />
                  <div>
                    <h2 className="text-2xl font-bold">Calendar</h2>
                    <p className="text-sm text-muted-foreground">{currentMonth}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setWeekOffset((prev) => prev - 1)}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setWeekOffset(0)}
                  >
                    Today
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setWeekOffset((prev) => prev + 1)}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-auto p-4">
            <div className="grid grid-cols-8 gap-2 min-h-full">
              <div className="flex flex-col pt-14">
                {Array.from({ length: 15 }, (_, i) => i + 8).map((hour) => (
                  <div key={hour} className="text-xs text-muted-foreground text-right pr-2 h-16 flex items-start">
                    {hour > 12 ? `${hour - 12}:00 PM` : hour === 12 ? "12:00 PM" : `${hour}:00 AM`}
                  </div>
                ))}
              </div>

              {weekDays.map((day, dayIndex) => {
                const date = weekDates[dayIndex];
                const dateNumber = date.getDate();
                const isToday = date.toDateString() === today.toDateString();
                const dayAppointments = appointmentsThisWeek.filter(
                  (apt) => (apt.start as Date).toDateString() === date.toDateString()
                );

                return (
                  <div key={day} className="flex flex-col">
                    <div className={`text-center pb-2 mb-2 border-b h-14 flex flex-col justify-center ${isToday ? "font-bold text-primary" : "text-muted-foreground"}`}>
                      <div className="text-xs mb-1">{day}</div>
                      <div className={`text-lg ${isToday ? "bg-primary text-primary-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto" : ""}`}>
                        {dateNumber}
                      </div>
                    </div>

                    <div className="flex-1 relative">
                      <div className="absolute inset-0 flex flex-col">
                        {Array.from({ length: 15 }).map((_, i) => (
                          <div key={i} className="h-16 border-t border-muted/30"></div>
                        ))}
                      </div>

                      {dayAppointments.map((apt) => {
                        const startDate = apt.start as Date;
                        const topMinutes = (startDate.getHours() - 8) * 60 + startDate.getMinutes();
                        const startOffset = (topMinutes / (15 * 60)) * 100;
                        const height = (apt.durationMinutes / (15 * 60)) * 100;

                        const endHour = startDate.getHours() + Math.floor(apt.durationMinutes / 60);
                        const endMinutes = (startDate.getMinutes() + apt.durationMinutes) % 60;
                        const endTimeLabel = `${endHour > 12 ? endHour - 12 : endHour}:${endMinutes
                          .toString()
                          .padStart(2, "0")} ${endHour >= 12 ? "PM" : "AM"}`;

                        const startLabel = startDate.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

                        return (
                          <div
                            key={apt.id}
                            className={`absolute ${apt.color} text-white rounded-lg p-2 left-0 right-0 mx-1 cursor-pointer hover:shadow-lg transition-shadow`}
                            style={{
                              top: `${startOffset}%`,
                              height: `${Math.max(height, 8)}%`,
                              minHeight: "50px",
                            }}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (apt.zoomLink) window.open(apt.zoomLink, "_blank", "noopener,noreferrer");
                            }}
                          >
                            <p className="text-xs font-semibold line-clamp-1">{apt.client}</p>
                            <p className="text-xs opacity-90">{startLabel} - {endTimeLabel}</p>
                            {apt.durationMinutes >= 30 && (
                              <p className="text-xs opacity-75 mt-1">{apt.title}</p>
                            )}
                          </div>
                        );
                      })}

                      {dayAppointments.length === 0 && (
                        <p className="text-xs text-muted-foreground px-2 py-4">No events</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
