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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect } from "react";
import { ChartPie, Users, Activity, ExternalLink, Database, Server } from "lucide-react"

// Grafana dashboard URLs based on environment
const getGrafanaUrl = () => {
  const isProduction = typeof window !== 'undefined' &&
    (window.location.hostname === 'bloomhealth.us' || window.location.hostname === 'www.bloomhealth.us');

  return isProduction
    ? 'https://opscvisuals.grafana.net/d/bloom-production/bloom-production'
    : 'https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa';
};

const getEnvironment = () => {
  if (typeof window === 'undefined') return 'Development';
  const hostname = window.location.hostname;
  if (hostname === 'bloomhealth.us' || hostname === 'www.bloomhealth.us') return 'Production';
  if (hostname === 'qa.bloomhealth.us') return 'QA';
  return 'Development';
};

export default function AdminPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  // Check if user is administrator
  const isAdmin = session?.user?.administrator === true;

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
                  <BreadcrumbItem className="hidden md:block">
                    <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator className="hidden md:block" />
                  <BreadcrumbItem>
                    <BreadcrumbLink href="/admin">Admin</BreadcrumbLink>
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
  const environment = getEnvironment();

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
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbLink href="/admin">Admin</BreadcrumbLink>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          {/* Stats Cards - Same grid as dashboard */}
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <Card className="aspect-video flex flex-col justify-between">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Observability</CardTitle>
                  <ChartPie className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent className="flex flex-col justify-end flex-1">
                <div className="text-2xl font-bold">Grafana</div>
                <p className="text-xs text-muted-foreground">Real-time metrics</p>
                <Button variant="outline" size="sm" className="mt-3" asChild>
                  <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                    Open <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </Button>
              </CardContent>
            </Card>

            <Card className="aspect-video flex flex-col justify-between">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Database</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent className="flex flex-col justify-end flex-1">
                <div className="text-2xl font-bold">CockroachDB</div>
                <p className="text-xs text-muted-foreground">
                  {environment === 'Production' ? 'meek-wallaby' : 'exotic-cuscus'}
                </p>
                <Button variant="outline" size="sm" className="mt-3" asChild>
                  <a href="https://cockroachlabs.cloud" target="_blank" rel="noopener noreferrer">
                    Console <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </Button>
              </CardContent>
            </Card>

            <Card className="aspect-video flex flex-col justify-between">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium">Hosting</CardTitle>
                  <Server className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent className="flex flex-col justify-end flex-1">
                <div className="text-2xl font-bold">Vercel</div>
                <p className="text-xs text-muted-foreground">{environment}</p>
                <Button variant="outline" size="sm" className="mt-3" asChild>
                  <a href="https://vercel.com/opsc/bloom" target="_blank" rel="noopener noreferrer">
                    Project <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Grafana Dashboard Embed - Main content area like dashboard skeleton */}
          <Card className="min-h-[100vh] flex-1 md:min-h-min">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Analytics Dashboard</CardTitle>
                  <CardDescription>{environment} Environment</CardDescription>
                </div>
                <Button size="sm" asChild>
                  <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                    Full Screen <ExternalLink className="ml-1 h-3 w-3" />
                  </a>
                </Button>
              </div>
            </CardHeader>
            <CardContent className="h-[calc(100vh-300px)] min-h-[500px]">
              <iframe
                src={`${grafanaUrl}?orgId=1&kiosk`}
                width="100%"
                height="100%"
                frameBorder="0"
                className="rounded-lg border"
                title="Grafana Dashboard"
              />
            </CardContent>
          </Card>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
