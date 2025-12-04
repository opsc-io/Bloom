"use client";

import { AppSidebar } from "@/components/app-sidebar"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar"
import { Search, Maximize2, Minimize2, MessageSquare, Send } from "lucide-react"

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
  const [searchQuery, setSearchQuery] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);

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
  // TODO: Replace with actual data from backend
  const mockMessages: Array<{
    id: number;
    sender: string;
    message: string;
    time: string;
    isMe: boolean;
    avatar: string;
    avatarColor: string;
  }> = [
    { id: 1, sender: "Dr. Sarah Johnson", message: "Hi, I have a question about the upcoming appointment.", time: "10:30 AM", isMe: false, avatar: "SJ", avatarColor: "bg-purple-500" },
    { id: 2, sender: "You", message: "Of course! How can I help you?", time: "10:32 AM", isMe: true, avatar: session?.user?.name?.[0] || "U", avatarColor: "bg-blue-500" },
    { id: 3, sender: "Dr. Sarah Johnson", message: "Can we reschedule to next Tuesday?", time: "10:33 AM", isMe: false, avatar: "SJ", avatarColor: "bg-purple-500" },
  ];

  // TODO: Replace with actual data from backend
  const mockConversations: Array<{
    id: number;
    name: string;
    avatar: string;
    avatarColor: string;
    lastMessage: string;
    time: string;
    unread: number;
    active: boolean;
  }> = [
    { id: 1, name: "Dr. Sarah Johnson", avatar: "SJ", avatarColor: "bg-purple-500", lastMessage: "Can we reschedule to next Tuesday?", time: "10:33 AM", unread: 1, active: true },
  ];



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
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4 justify-between">
          <div className="flex items-center gap-2">
            <SidebarTrigger className="-ml-1" />
            <Separator
              orientation="vertical"
              className="mr-2 data-[orientation=vertical]:h-4"
            />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink href="#">
                    Building Your Application
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  <BreadcrumbPage>Data Fetching</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative w-64">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search therapists..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-3 p-4 pt-4">
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            {/* Chat Messages Preview Box */}
            <Card 
              className={`transition-all duration-300 ${
                isExpanded 
                  ? 'fixed inset-0 z-50 rounded-none border-0 md:col-span-1' 
                  : 'md:col-span-1 aspect-video cursor-pointer hover:shadow-lg'
              }`}
              onClick={() => !isExpanded && setIsExpanded(true)}
            >
              {!isExpanded ? (
                // Collapsed Preview View
                <>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <MessageSquare className="h-5 w-5 text-primary" />
                        <CardTitle>Messages</CardTitle>
                      </div>
                      <Maximize2 className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <CardDescription>Recent conversations</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-3 flex-wrap">
                      {mockConversations.length > 0 ? (
                        mockConversations.map((conv) => (
                          <div key={conv.id} className="relative">
                            <Avatar className="h-12 w-12 cursor-pointer hover:scale-110 transition-transform">
                              <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                                {conv.avatar}
                              </AvatarFallback>
                            </Avatar>
                            {conv.unread > 0 && (
                              <span className="absolute -top-1 -right-1 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center font-semibold border-2 border-background">
                                {conv.unread}
                              </span>
                            )}
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-muted-foreground py-4">No messages yet</p>
                      )}
                    </div>
                  </CardContent>
                </>
              ) : (
                // Expanded Full-Screen Chat View
                <div className="flex h-screen">
                  {/* Left Sidebar - Conversations List */}
                  <div className="w-80 border-r bg-muted/30 flex flex-col">
                    <div className="p-4 border-b">
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-bold">Chats</h2>
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
                      <div className="relative">
                        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input placeholder="Search conversations..." className="pl-8" />
                      </div>
                    </div>
                    <div className="flex-1 overflow-y-auto">
                      {mockConversations.length > 0 ? (
                        mockConversations.map((conv) => (
                          <div
                            key={conv.id}
                            className={`flex items-center gap-3 p-4 hover:bg-muted/50 cursor-pointer transition-colors ${
                              conv.active ? 'bg-muted/50 border-l-2 border-primary' : ''
                            }`}
                          >
                            <Avatar className="h-12 w-12 shrink-0">
                              <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                                {conv.avatar}
                              </AvatarFallback>
                            </Avatar>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between mb-1">
                                <p className="font-semibold text-sm truncate">{conv.name}</p>
                                <span className="text-xs text-muted-foreground">{conv.time}</span>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-muted-foreground truncate">{conv.lastMessage}</p>
                                {conv.unread > 0 && (
                                  <span className="ml-2 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center shrink-0">
                                    {conv.unread}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-muted-foreground">No conversations yet</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right Side - Active Conversation */}
                  <div className="flex-1 flex flex-col">
                    {/* Chat Header */}
                    <div className="p-4 border-b bg-background">
                      {mockConversations.find(c => c.active) ? (
                        <div className="flex items-center gap-3">
                          <Avatar className="h-10 w-10">
                            <AvatarFallback className={`${mockConversations.find(c => c.active)?.avatarColor} text-white font-semibold`}>
                              {mockConversations.find(c => c.active)?.avatar}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-semibold">{mockConversations.find(c => c.active)?.name}</p>
                            <p className="text-xs text-muted-foreground">Active now</p>
                          </div>
                        </div>
                      ) : (
                        <p className="text-muted-foreground">Select a conversation</p>
                      )}
                    </div>

                    {/* Messages Area */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-muted/20">
                      {mockMessages.length > 0 ? (
                        mockMessages.map((msg) => (
                          <div
                            key={msg.id}
                            className={`flex gap-3 ${msg.isMe ? 'justify-end' : 'justify-start'}`}
                          >
                            {!msg.isMe && (
                              <Avatar className="h-8 w-8 shrink-0">
                                <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                                  {msg.avatar}
                                </AvatarFallback>
                              </Avatar>
                            )}
                            <div className={`flex flex-col ${msg.isMe ? 'items-end' : 'items-start'} max-w-[60%]`}>
                              {!msg.isMe && <p className="text-xs font-semibold text-muted-foreground mb-1">{msg.sender}</p>}
                              <div className={`${msg.isMe ? 'bg-primary text-primary-foreground' : 'bg-background border'} rounded-2xl px-4 py-2.5 shadow-sm`}>
                                <p className="text-sm">{msg.message}</p>
                              </div>
                              <p className="text-xs text-muted-foreground mt-1">{msg.time}</p>
                            </div>
                            {msg.isMe && (
                              <Avatar className="h-8 w-8 shrink-0">
                                <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                                  {msg.avatar}
                                </AvatarFallback>
                              </Avatar>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-muted-foreground">No messages yet</p>
                        </div>
                      )}
                    </div>

                    {/* Message Input Area */}
                    <div className="p-4 border-t bg-background">
                      <div className="flex gap-2">
                        <Input placeholder="Type a message..." className="flex-1" />
                        <Button size="icon" className="shrink-0">
                          <Send className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Card>
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
