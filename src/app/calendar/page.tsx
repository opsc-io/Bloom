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
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Clock, Edit3, XCircle } from "lucide-react";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Appointment = {
  id: string;
  title: string;
  start: string | Date;
  end: string | Date;
  durationMinutes: number;
  client: string;
  color: string;
  zoomLink?: string;
  status?: string;
  therapistId?: string;
  patientId?: string;
};

export default function CalendarPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [weekOffset, setWeekOffset] = useState(0);
  const [connections, setConnections] = useState<Array<{ id: string; name: string }>>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [selectedParticipant, setSelectedParticipant] = useState<string>("");
  const [date, setDate] = useState<string>("");
  const [startTime, setStartTime] = useState<string>("10:00");
  const [duration, setDuration] = useState<number>(30);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);
  const [editDate, setEditDate] = useState<string>("");
  const [editStartTime, setEditStartTime] = useState<string>("");
  const [editDuration, setEditDuration] = useState<number>(30);
  const [editError, setEditError] = useState<string | null>(null);
  const [editSaving, setEditSaving] = useState(false);

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

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;
    const loadConnections = async () => {
      try {
        const res = await fetch("/api/user/connections");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setConnections(
          (data.people ?? []).map((p: any) => ({
            id: p.id,
            name: p.name ?? p.email ?? "Unknown",
          }))
        );
      } catch {
        // ignore
      }
    };
    loadConnections();
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
  const isTherapist = (user as { role?: string }).role === "THERAPIST";

  const handleCreateAppointment = async () => {
    if (!selectedParticipant || !date || !startTime || duration <= 0) {
      setError("Please fill all fields");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const [hours, minutes] = startTime.split(":").map((v) => parseInt(v, 10));
      const start = new Date(date);
      start.setHours(hours, minutes, 0, 0);
      const end = new Date(start.getTime() + duration * 60000);

      const res = await fetch("/api/appointments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          startAt: start.toISOString(),
          endAt: end.toISOString(),
          participantId: selectedParticipant,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to create appointment");
      }

      // refresh appointments
      const refresh = await fetch("/api/appointments");
      if (refresh.ok) {
        const data = (await refresh.json()) as { appointments?: Appointment[] };
        setAppointments(
          (data.appointments ?? []).map((apt) => ({
            ...apt,
            start: new Date(apt.start),
            end: new Date(apt.end),
          }))
        );
      }
      setShowCreate(false);
      setSelectedParticipant("");
      setDate("");
      setStartTime("10:00");
      setDuration(30);
    } catch (err: any) {
      setError(err.message || "Unable to create appointment");
    } finally {
      setSaving(false);
    }
  };

  const handleSelectAppointment = (apt: Appointment) => {
    setSelectedAppointment(apt);
    const startDate = apt.start as Date;
    setEditDate(startDate.toISOString().slice(0, 10));
    setEditStartTime(startDate.toISOString().slice(11, 16));
    setEditDuration(apt.durationMinutes);
    setEditError(null);
  };

  const handleEditAppointment = async () => {
    if (!selectedAppointment) return;
    if (!editDate || !editStartTime || editDuration <= 0) {
      setEditError("Please fill all fields");
      return;
    }
    setEditSaving(true);
    setEditError(null);
    try {
      const [hours, minutes] = editStartTime.split(":").map((v) => parseInt(v, 10));
      const start = new Date(editDate);
      start.setHours(hours, minutes, 0, 0);
      const end = new Date(start.getTime() + editDuration * 60000);

      const res = await fetch("/api/appointments", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          appointmentId: selectedAppointment.id,
          startAt: start.toISOString(),
          endAt: end.toISOString(),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to update appointment");
      }

      const refresh = await fetch("/api/appointments");
      if (refresh.ok) {
        const data = (await refresh.json()) as { appointments?: Appointment[] };
        setAppointments(
          (data.appointments ?? []).map((apt) => ({
            ...apt,
            start: new Date(apt.start),
            end: new Date(apt.end),
          }))
        );
      }
      setSelectedAppointment(null);
    } catch (err: any) {
      setEditError(err.message || "Unable to update appointment");
    } finally {
      setEditSaving(false);
    }
  };

  const handleCancelAppointment = async () => {
    if (!selectedAppointment) return;
    setEditSaving(true);
    setEditError(null);
    try {
      const res = await fetch("/api/appointments", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ appointmentId: selectedAppointment.id }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to cancel appointment");
      }

      const refresh = await fetch("/api/appointments");
      if (refresh.ok) {
        const data = (await refresh.json()) as { appointments?: Appointment[] };
        setAppointments(
          (data.appointments ?? []).map((apt) => ({
            ...apt,
            start: new Date(apt.start),
            end: new Date(apt.end),
          }))
        );
      }
      setSelectedAppointment(null);
    } catch (err: any) {
      setEditError(err.message || "Unable to cancel appointment");
    } finally {
      setEditSaving(false);
    }
  };

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
                {connections.length > 0 && (
                  <Button size="sm" onClick={() => setShowCreate(true)}>
                    New Appointment
                  </Button>
                )}
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
                              handleSelectAppointment(apt);
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
          {showCreate && (
            <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50">
              <div className="bg-background w-full max-w-lg rounded-lg shadow-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Create appointment</h3>
                    <p className="text-sm text-muted-foreground">
                      {isTherapist ? "Select a patient and time" : "Select your therapist and time"}
                    </p>
                  </div>
                  <Button variant="ghost" onClick={() => setShowCreate(false)}>
                    Close
                  </Button>
                </div>

                <div className="space-y-3">
                  <div className="space-y-1">
                    <Label>Participant</Label>
                    <select
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      value={selectedParticipant}
                      onChange={(e) => setSelectedParticipant(e.target.value)}
                    >
                      <option value="">{isTherapist ? "Choose patient" : "Choose therapist"}</option>
                      {connections.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label>Date</Label>
                      <Input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
                    </div>
                    <div className="space-y-1">
                      <Label>Start time</Label>
                      <Input type="time" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
                    </div>
                  </div>
                  <div className="space-y-1">
                    <Label>Duration (minutes)</Label>
                    <Input
                      type="number"
                      min={15}
                      step={15}
                      value={duration}
                      onChange={(e) => setDuration(Number(e.target.value) || 0)}
                    />
                  </div>
                  {error && <p className="text-sm text-destructive">{error}</p>}
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <Button variant="ghost" onClick={() => setShowCreate(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateAppointment} disabled={saving}>
                    {saving ? "Saving..." : "Create"}
                  </Button>
                </div>
              </div>
            </div>
          )}
          {selectedAppointment && (
            <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50">
              <div className="bg-background w-full max-w-lg rounded-lg shadow-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Appointment</h3>
                    <p className="text-sm text-muted-foreground">
                      {selectedAppointment.client} • {selectedAppointment.status ?? "SCHEDULED"}
                    </p>
                  </div>
                  <Button variant="ghost" onClick={() => setSelectedAppointment(null)}>
                    Close
                  </Button>
                </div>

                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label>Date</Label>
                      <Input type="date" value={editDate} onChange={(e) => setEditDate(e.target.value)} disabled={!isTherapist} />
                    </div>
                    <div className="space-y-1">
                      <Label>Start time</Label>
                      <Input type="time" value={editStartTime} onChange={(e) => setEditStartTime(e.target.value)} disabled={!isTherapist} />
                    </div>
                  </div>
                  <div className="space-y-1">
                    <Label>Duration (minutes)</Label>
                    <Input
                      type="number"
                      min={15}
                      step={15}
                      value={editDuration}
                      onChange={(e) => setEditDuration(Number(e.target.value) || 0)}
                      disabled={!isTherapist}
                    />
                  </div>
                  {editError && <p className="text-sm text-destructive">{editError}</p>}
                </div>

                <div className="flex justify-between pt-2">
                  <div>
                    {!isTherapist && selectedAppointment.patientId === user.id && (
                      <Button variant="destructive" onClick={handleCancelAppointment} disabled={editSaving}>
                        {editSaving ? "Cancelling..." : <><XCircle className="h-4 w-4 mr-2" />Cancel</>}
                      </Button>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="ghost" onClick={() => setSelectedAppointment(null)} disabled={editSaving}>
                      Close
                    </Button>
                    {isTherapist && selectedAppointment.therapistId === user.id && (
                      <Button onClick={handleEditAppointment} disabled={editSaving}>
                        {editSaving ? "Saving..." : <><Edit3 className="h-4 w-4 mr-2" />Save</>}
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
