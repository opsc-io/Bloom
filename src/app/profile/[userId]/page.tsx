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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { use, useEffect, useState } from "react";
import { Mail, Calendar, MapPin, Briefcase, MessageCircle } from "lucide-react";

interface UserProfile {
  id: string;
  firstname: string;
  lastname: string;
  email: string;
  image?: string | null;
  role?: string;
  createdAt: string;
}

export default function ProfileViewPage({ params }: { params: Promise<{ userId: string }> }) {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const { userId } = use(params);

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await fetch(`/api/user/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setProfile(data);
        }
      } catch (error) {
        console.error("Error fetching profile:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [userId]);

  if (isPending || loading)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;
  const displayName = profile ? `${profile.firstname} ${profile.lastname}` : "User";
  const memberSince = profile?.createdAt ? new Date(profile.createdAt).toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) : '';

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
                <BreadcrumbPage>Profile</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-6">
          <div className="mx-auto w-full max-w-4xl">
            {!profile ? (
              <Card>
                <CardContent className="flex items-center justify-center py-12">
                  <p className="text-muted-foreground">Profile not found</p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-6">
                {/* Profile Header Card */}
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex flex-col items-center text-center space-y-4">
                      <Avatar className="h-24 w-24">
                        <AvatarImage src={profile.image || ""} />
                        <AvatarFallback className="text-2xl">
                          {profile.firstname?.[0]}{profile.lastname?.[0]}
                        </AvatarFallback>
                      </Avatar>
                      
                      <div className="space-y-2">
                        <h1 className="text-3xl font-bold">{displayName}</h1>
                        {profile.role && profile.role !== "UNSET" && (
                          <Badge variant="secondary" className="text-sm">
                            {profile.role === "THERAPIST" ? "Therapist" : "Patient"}
                          </Badge>
                        )}
                      </div>

                      <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <Mail className="h-4 w-4" />
                          <span>{profile.email}</span>
                        </div>
                        {memberSince && (
                          <div className="flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            <span>Member since {memberSince}</span>
                          </div>
                        )}
                      </div>

                      {/* Message Button - Only show if viewing someone else's profile */}
                      {session?.user?.id !== userId && (
                        <Button 
                          onClick={() => router.push(`/dashboard?message=${userId}`)}
                          className="mt-2"
                        >
                          <MessageCircle className="h-4 w-4 mr-2" />
                          Message Me
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* About Section */}
                <Card>
                  <CardHeader>
                    <CardTitle>About</CardTitle>
                    <CardDescription>Profile information</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-3">
                      <div className="flex items-start gap-3">
                        <Briefcase className="h-5 w-5 text-muted-foreground mt-0.5" />
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Role</p>
                          <p className="text-sm text-muted-foreground">
                            {profile.role === "THERAPIST" ? "Licensed Therapist" : profile.role === "PATIENT" ? "Patient" : "Not set"}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-start gap-3">
                        <Mail className="h-5 w-5 text-muted-foreground mt-0.5" />
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Email</p>
                          <p className="text-sm text-muted-foreground">{profile.email}</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
