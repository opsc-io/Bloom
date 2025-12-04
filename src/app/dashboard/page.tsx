"use client";

import { AppSidebar } from "@/components/app-sidebar"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  //BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"


import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";



export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [roleDialogOpen, setRoleDialogOpen] = useState(false);
  const [roleError, setRoleError] = useState<string | null>(null);
  const [isSavingRole, setIsSavingRole] = useState(false);
  const [roleAcknowledged, setRoleAcknowledged] = useState(false);

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
    if (shouldPromptForRole && !roleAcknowledged) {
      setRoleDialogOpen(true);
    }
  }, [shouldPromptForRole, roleAcknowledged]);

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
        setRoleAcknowledged(true);
        router.refresh();
      }
    } catch (err) {
      setRoleError("Unable to save your role. Please try again.");
    } finally {
      setIsSavingRole(false);
    }
  }

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
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
            <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
            <Skeleton className="bg-muted/50 aspect-video rounded-xl" />
          </div>
          <Skeleton className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" />
        </div>
      </SidebarInset>
      {roleDialogOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-md rounded-lg bg-background p-6 shadow-lg">
            <h2 className="text-lg font-semibold">Tell us about you</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Are you using Bloom as a practitioner or a patient? We use this to tailor your dashboard.
            </p>
            {roleError ? (
              <p className="mt-3 text-sm text-red-500">{roleError}</p>
            ) : null}
            <div className="mt-5 flex flex-col gap-3 sm:flex-row">
              <Button
                onClick={() => handleRoleSelect("practitioner")}
                disabled={isSavingRole}
              >
                I am a Practitioner
              </Button>
              <Button
                variant="outline"
                onClick={() => handleRoleSelect("patient")}
                disabled={isSavingRole}
              >
                I am a Patient
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </SidebarProvider>
  )
}
