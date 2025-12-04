"use client";

import { AppSidebar } from "@/components/app-sidebar"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { DashboardHeader } from "@/components/dashboard-header";
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useMemo, useState } from "react";
import { DashboardRoleDialog } from "@/components/dashboard-role-dialog";
import { DashboardMessagingCard } from "@/components/dashboard-messaging-card";
import { DashboardCalendarCard } from "@/components/dashboard-calendar-card";

type Appointment = {
  id: string;
  title: string;
  start: string;
  end: string;
  durationMinutes: number;
  client: string;
  color: string;
  zoomLink?: string;
};

export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [roleDialogOpen, setRoleDialogOpen] = useState(false);
  const [roleError, setRoleError] = useState<string | null>(null);
  const [isSavingRole, setIsSavingRole] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

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
  const [appointments, setAppointments] = useState<Appointment[]>([]);

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
            start: apt.start,
            end: apt.end,
          }))
        );
      } catch {
        // keep dashboard usable even if appointments fail
      }
    };

    loadAppointments();
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
      <AppSidebar user={user} />
      <SidebarInset>
        <DashboardHeader searchQuery={searchQuery} onSearchChange={setSearchQuery} />
        <div className="flex flex-1 flex-col gap-3 p-4 pt-4">
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <DashboardMessagingCard
              conversations={conversations}
              messages={messages}
            />
            <DashboardCalendarCard appointments={appointments} />
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
