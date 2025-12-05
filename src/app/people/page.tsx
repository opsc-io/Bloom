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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Users, User, MessageCircle, Mail, Search, UserPlus } from "lucide-react";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";

type Person = {
  id: string;
  name: string;
  email: string;
  image?: string | null;
  role?: string;
};

export default function PeoplePage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [people, setPeople] = useState<Person[]>([]);
  const [availableTherapists, setAvailableTherapists] = useState<Person[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [therapistSearchQuery, setTherapistSearchQuery] = useState("");

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;

    const loadPeople = async () => {
      try {
        const res = await fetch("/api/user/connections");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setPeople(data.people ?? []);
      } catch {
        // swallow errors to keep page responsive
      }
    };

    loadPeople();
    return () => {
      cancelled = true;
    };
  }, [isPending, session]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    const userRole = ((session.user as { role?: string }).role || "UNSET");
    if (userRole !== "PATIENT") return;
    
    let cancelled = false;

    const loadAvailableTherapists = async () => {
      try {
        const res = await fetch("/api/therapists/available");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setAvailableTherapists(data.therapists ?? []);
      } catch {
        // swallow errors to keep page responsive
      }
    };

    loadAvailableTherapists();
    return () => {
      cancelled = true;
    };
  }, [isPending, session]);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;
  const userRole = (user as { role?: string }).role || "UNSET";
  const isTherapist = userRole === "THERAPIST";

  const filteredPeople = people.filter((person) =>
    person.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    person.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredTherapists = availableTherapists.filter((therapist) =>
    therapist.name.toLowerCase().includes(therapistSearchQuery.toLowerCase()) ||
    therapist.email.toLowerCase().includes(therapistSearchQuery.toLowerCase())
  );

  const pageTitle = isTherapist ? "My Patients" : "My Therapist";
  const pageDescription = isTherapist
    ? "Manage and communicate with your patients"
    : "View and contact your assigned therapist";
  const Icon = isTherapist ? Users : User;

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
                <BreadcrumbPage>People</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-4">
          <Card className="bg-card/50 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-2xl font-bold flex items-center gap-2">
                    <Icon className="h-6 w-6" />
                    {pageTitle}
                  </CardTitle>
                  <CardDescription>{pageDescription}</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {!isTherapist ? (
                <Tabs defaultValue="my-therapist" className="w-full">
                  <TabsList className="grid w-full grid-cols-2 mb-4">
                    <TabsTrigger value="my-therapist">My Therapist</TabsTrigger>
                    <TabsTrigger value="find-therapist">Find Therapist</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="my-therapist" className="space-y-4">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search therapist..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 bg-muted/50"
                      />
                    </div>

                    {filteredPeople.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center">
                        <User className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-lg font-medium text-muted-foreground">
                          {searchQuery ? "No results found" : "No therapist assigned yet"}
                        </p>
                        {searchQuery && (
                          <p className="text-sm text-muted-foreground mt-2">
                            Try adjusting your search terms
                          </p>
                        )}
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredPeople.map((person) => (
                          <Card
                            key={person.id}
                            className="bg-muted/30 hover:bg-muted/50 transition-all hover:shadow-md"
                          >
                            <CardContent className="p-6">
                              <div className="flex items-start gap-4">
                                <Avatar className="h-16 w-16">
                                  <AvatarImage src={person.image || undefined} />
                                  <AvatarFallback className="text-lg">
                                    {person.name
                                      .split(" ")
                                      .map((n) => n[0])
                                      .join("")
                                      .toUpperCase()}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 min-w-0">
                                  <h3 className="font-semibold text-lg truncate">
                                    {person.name}
                                  </h3>
                                  <div className="flex items-center gap-1 text-sm text-muted-foreground mt-1">
                                    <Mail className="h-3 w-3" />
                                    <p className="truncate">{person.email}</p>
                                  </div>
                                  {person.role && (
                                    <p className="text-xs text-muted-foreground mt-2 capitalize">
                                      {person.role.toLowerCase()}
                                    </p>
                                  )}
                                </div>
                              </div>
                              <div className="flex gap-2 mt-4">
                                <Button
                                  variant="default"
                                  size="sm"
                                  className="flex-1"
                                  onClick={() => router.push(`/messages?message=${person.id}`)}
                                >
                                  <MessageCircle className="h-4 w-4 mr-2" />
                                  Message
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => router.push(`/profile/${person.id}`)}
                                >
                                  View Profile
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="find-therapist" className="space-y-4">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search available therapists..."
                        value={therapistSearchQuery}
                        onChange={(e) => setTherapistSearchQuery(e.target.value)}
                        className="pl-9 bg-muted/50"
                      />
                    </div>

                    {filteredTherapists.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center">
                        <UserPlus className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-lg font-medium text-muted-foreground">
                          {therapistSearchQuery ? "No therapists found" : "No available therapists"}
                        </p>
                        {therapistSearchQuery && (
                          <p className="text-sm text-muted-foreground mt-2">
                            Try adjusting your search terms
                          </p>
                        )}
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredTherapists.map((therapist) => (
                          <Card
                            key={therapist.id}
                            className="bg-muted/30 hover:bg-muted/50 transition-all hover:shadow-md"
                          >
                            <CardContent className="p-6">
                              <div className="flex items-start gap-4">
                                <Avatar className="h-16 w-16">
                                  <AvatarImage src={therapist.image || undefined} />
                                  <AvatarFallback className="text-lg">
                                    {therapist.name
                                      .split(" ")
                                      .map((n) => n[0])
                                      .join("")
                                      .toUpperCase()}
                                  </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 min-w-0">
                                  <h3 className="font-semibold text-lg truncate">
                                    {therapist.name}
                                  </h3>
                                  <div className="flex items-center gap-1 text-sm text-muted-foreground mt-1">
                                    <Mail className="h-3 w-3" />
                                    <p className="truncate">{therapist.email}</p>
                                  </div>
                                  <p className="text-xs text-muted-foreground mt-2">
                                    Therapist
                                  </p>
                                </div>
                              </div>
                              <div className="flex gap-2 mt-4">
                                <Button
                                  variant="default"
                                  size="sm"
                                  className="flex-1"
                                  onClick={() => router.push(`/messages?message=${therapist.id}`)}
                                >
                                  <MessageCircle className="h-4 w-4 mr-2" />
                                  Message
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => router.push(`/profile/${therapist.id}`)}
                                >
                                  View Profile
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              ) : (
                <>
                  <div className="mb-4">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search patients..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 bg-muted/50"
                      />
                    </div>
                  </div>

                  {filteredPeople.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <Users className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-lg font-medium text-muted-foreground">
                        {searchQuery ? "No results found" : "No patients assigned yet"}
                      </p>
                      {searchQuery && (
                        <p className="text-sm text-muted-foreground mt-2">
                          Try adjusting your search terms
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {filteredPeople.map((person) => (
                        <Card
                          key={person.id}
                          className="bg-muted/30 hover:bg-muted/50 transition-all hover:shadow-md"
                        >
                          <CardContent className="p-6">
                            <div className="flex items-start gap-4">
                              <Avatar className="h-16 w-16">
                                <AvatarImage src={person.image || undefined} />
                                <AvatarFallback className="text-lg">
                                  {person.name
                                    .split(" ")
                                    .map((n) => n[0])
                                    .join("")
                                    .toUpperCase()}
                                </AvatarFallback>
                              </Avatar>
                              <div className="flex-1 min-w-0">
                                <h3 className="font-semibold text-lg truncate">
                                  {person.name}
                                </h3>
                                <div className="flex items-center gap-1 text-sm text-muted-foreground mt-1">
                                  <Mail className="h-3 w-3" />
                                  <p className="truncate">{person.email}</p>
                                </div>
                                {person.role && (
                                  <p className="text-xs text-muted-foreground mt-2 capitalize">
                                    {person.role.toLowerCase()}
                                  </p>
                                )}
                              </div>
                            </div>
                            <div className="flex gap-2 mt-4">
                              <Button
                                variant="default"
                                size="sm"
                                className="flex-1"
                                onClick={() => router.push(`/messages?message=${person.id}`)}
                              >
                                <MessageCircle className="h-4 w-4 mr-2" />
                                Message
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => router.push(`/profile/${person.id}`)}
                              >
                                View Profile
                              </Button>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
