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
import { ChartPie, Users, Activity, ExternalLink } from "lucide-react"

// Grafana dashboard URLs based on environment
const getGrafanaUrl = () => {
  const isProduction = typeof window !== 'undefined' &&
    (window.location.hostname === 'bloomhealth.us' || window.location.hostname === 'www.bloomhealth.us');

  return isProduction
    ? 'https://opscvisuals.grafana.net/d/bloom-production/bloom-production'
    : 'https://opscvisuals.grafana.net/d/bloom-qa/bloom-qa';
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
  const environment = typeof window !== 'undefined' &&
    (window.location.hostname === 'bloomhealth.us' || window.location.hostname === 'www.bloomhealth.us')
    ? 'Production'
    : 'QA/Development';

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
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Admin Dashboard</h1>
              <p className="text-muted-foreground">
                Environment: <span className="font-medium">{environment}</span>
              </p>
            </div>
            <Button asChild>
              <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                Open Full Dashboard <ExternalLink className="ml-2 h-4 w-4" />
              </a>
            </Button>
          </div>

          {/* Quick Stats Cards */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Observability</CardTitle>
                <ChartPie className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">Grafana</div>
                <p className="text-xs text-muted-foreground">
                  Real-time metrics and analytics
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Database</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">CockroachDB</div>
                <p className="text-xs text-muted-foreground">
                  User and session data
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Hosting</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">Vercel</div>
                <p className="text-xs text-muted-foreground">
                  Deployment and logs
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Grafana Embed */}
          <Card className="flex-1">
            <CardHeader>
              <CardTitle>Analytics Dashboard</CardTitle>
              <CardDescription>
                Live metrics from Grafana Cloud - {environment} environment
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[600px]">
              <iframe
                src={`${grafanaUrl}?orgId=1&kiosk`}
                width="100%"
                height="100%"
                frameBorder="0"
                className="rounded-lg"
                title="Grafana Dashboard"
              />
            </CardContent>
          </Card>

          {/* Quick Links */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Links</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                <Button variant="outline" asChild className="justify-start">
                  <a href={grafanaUrl} target="_blank" rel="noopener noreferrer">
                    <ChartPie className="mr-2 h-4 w-4" /> Grafana Dashboard
                  </a>
                </Button>
                <Button variant="outline" asChild className="justify-start">
                  <a href="https://vercel.com/opsc/bloom" target="_blank" rel="noopener noreferrer">
                    <Activity className="mr-2 h-4 w-4" /> Vercel Project
                  </a>
                </Button>
                <Button variant="outline" asChild className="justify-start">
                  <a href="https://cockroachlabs.cloud" target="_blank" rel="noopener noreferrer">
                    <Users className="mr-2 h-4 w-4" /> CockroachDB Console
                  </a>
                </Button>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Environment Info</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Environment</dt>
                    <dd className="font-medium">{environment}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Database</dt>
                    <dd className="font-medium">{environment === 'Production' ? 'meek-wallaby' : 'exotic-cuscus'}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Domain</dt>
                    <dd className="font-medium">{typeof window !== 'undefined' ? window.location.hostname : 'localhost'}</dd>
                  </div>
                </dl>
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
