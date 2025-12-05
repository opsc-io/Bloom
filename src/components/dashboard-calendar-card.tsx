import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
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
  const router = useRouter();
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
      className="md:col-span-1 h-full cursor-pointer hover:shadow-lg transition-all"
      onClick={() => router.push('/calendar')}
    >
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
    </Card>
  );
}
