import { useMemo, useState } from "react";
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Clock, Maximize2, Minimize2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

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

type Props = {
  appointments: Appointment[];
};

export function DashboardCalendarCard({ appointments }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
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

  const appointmentsThisWeek = useMemo(() => {
    return appointments
      .map((apt) => ({
        ...apt,
        start: new Date(apt.start),
        end: new Date(apt.end),
      }))
      .filter((apt) => {
        const start = apt.start as Date;
        return start >= weekStart && start < new Date(weekStart.getTime() + 7 * 24 * 60 * 60 * 1000);
      })
      .sort((a, b) => (a.start as Date).getTime() - (b.start as Date).getTime());
  }, [appointments, weekStart]);

  const todaysAppointments = appointmentsThisWeek.filter(
    (apt) => (apt.start as Date).toDateString() === today.toDateString()
  );

  const currentMonth = weekStart.toLocaleDateString("en-US", { month: "long", year: "numeric" });

  return (
    <Card
      className={`transition-all duration-300 ${isExpanded
        ? "fixed inset-0 z-50 rounded-none border-0 md:col-span-1"
        : "md:col-span-1 aspect-video cursor-pointer hover:shadow-lg"
        }`}
      onClick={() => !isExpanded && setIsExpanded(true)}
    >
      {!isExpanded ? (
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
              {today.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}
            </CardDescription>
            <p className="text-xs text-muted-foreground mt-1">
              {todaysAppointments.length} {todaysAppointments.length === 1 ? "appointment" : "appointments"} today
            </p>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden pb-4">
            <div className="space-y-1.5 h-full flex flex-col">
              {todaysAppointments.slice(0, 2).map((apt) => {
                const start = apt.start as Date;
                const time = start.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
                return (
                  <div key={apt.id} className="flex items-center gap-3 p-1.5 rounded-lg hover:bg-muted/50 transition-colors">
                    <div className={`h-8 w-1 rounded-full ${apt.color}`}></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold truncate">{apt.client}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span>{time} • {apt.durationMinutes}m</span>
                      </div>
                    </div>
                  </div>
                );
              })}
              {todaysAppointments.length > 2 && (
                <div className="flex pl-2 pt-0.5">
                  <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                    +{todaysAppointments.length - 2} more
                  </span>
                </div>
              )}
              {todaysAppointments.length === 0 && (
                <p className="text-sm text-muted-foreground py-4">No appointments today</p>
              )}
            </div>
          </CardContent>
        </>
      ) : (
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
                      setWeekOffset((prev) => prev - 1);
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
                      setWeekOffset((prev) => prev + 1);
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
                  setIsExpanded(false);
                }}
              >
                <Minimize2 className="h-4 w-4" />
              </Button>
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
      )}
    </Card>
  );
}
