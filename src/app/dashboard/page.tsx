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
import { Search, Maximize2, Minimize2, MessageSquare, Send, Calendar as CalendarIcon, Clock, ChevronLeft, ChevronRight } from "lucide-react"

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";



export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [searchQuery, setSearchQuery] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);
  const [isCalendarExpanded, setIsCalendarExpanded] = useState(false);
  const [weekOffset, setWeekOffset] = useState(0);

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

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

  // TODO: Replace with actual data from backend
  const mockAppointments: Array<{
    id: number;
    title: string;
    time: string;
    duration: string;
    client: string;
    color: string;
    startHour: number;
    durationMinutes: number;
    zoomLink?: string;
  }> = [
    { id: 1, title: "Therapy Session", time: "09:00 AM", duration: "1h", client: "Sarah Johnson", color: "bg-blue-500", startHour: 9, durationMinutes: 60, zoomLink: "https://zoom.us/j/1234567890" },
    { id: 2, title: "Initial Consultation", time: "11:00 AM", duration: "45m", client: "Michael Chen", color: "bg-emerald-500", startHour: 11, durationMinutes: 45, zoomLink: "https://zoom.us/j/0987654321" },
    { id: 3, title: "Follow-up", time: "02:00 PM", duration: "30m", client: "Emily Roberts", color: "bg-purple-500", startHour: 14, durationMinutes: 30, zoomLink: "https://zoom.us/j/1122334455" },
  ];

  const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const today = new Date();
  
  // Calculate the week start based on offset
  const getWeekStart = (offset: number) => {
    const date = new Date(today);
    const dayOfWeek = date.getDay();
    const diff = dayOfWeek === 0 ? -6 : 1 - dayOfWeek; // Adjust to Monday
    date.setDate(date.getDate() + diff + (offset * 7));
    return date;
  };
  
  const weekStart = getWeekStart(weekOffset);
  const currentMonth = weekStart.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  const currentDay = today.toLocaleDateString('en-US', { weekday: 'short' });
  
  // Generate week dates
  const weekDates = weekDays.map((_, index) => {
    const date = new Date(weekStart);
    date.setDate(weekStart.getDate() + index);
    return date;
  });


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
            
            {/* Calendar Preview Box */}
            <Card 
              className={`transition-all duration-300 ${
                isCalendarExpanded 
                  ? 'fixed inset-0 z-50 rounded-none border-0 md:col-span-1' 
                  : 'md:col-span-1 aspect-video cursor-pointer hover:shadow-lg'
              }`}
              onClick={() => !isCalendarExpanded && setIsCalendarExpanded(true)}
            >
              {!isCalendarExpanded ? (
                // Collapsed Preview View
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
                      {today.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                    </CardDescription>
                    <p className="text-xs text-muted-foreground mt-1">
                      {mockAppointments.length} {mockAppointments.length === 1 ? 'appointment' : 'appointments'} today
                    </p>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-hidden pb-4">
                    <div className="space-y-1.5 h-full flex flex-col">
                      {mockAppointments.slice(0, 2).map((apt) => (
                        <div key={apt.id} className="flex items-center gap-3 p-1.5 rounded-lg hover:bg-muted/50 transition-colors">
                          <div className={`h-8 w-1 rounded-full ${apt.color}`}></div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold truncate">{apt.client}</p>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span>{apt.time} • {apt.duration}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                      {mockAppointments.length > 2 && (
                        <div className="flex pl-2 pt-0.5">
                          <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
                            +{mockAppointments.length - 2} more
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </>
              ) : (
                // Expanded Weekly Calendar View
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
                              setWeekOffset(prev => prev - 1);
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
                              setWeekOffset(prev => prev + 1);
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
                          setIsCalendarExpanded(false);
                        }}
                      >
                        <Minimize2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  
                  {/* Weekly Calendar Grid */}
                  <div className="flex-1 overflow-auto p-4">
                    <div className="grid grid-cols-8 gap-2 min-h-full">
                      {/* Time column */}
                      <div className="flex flex-col pt-14">
                        {Array.from({ length: 15 }, (_, i) => i + 8).map((hour) => (
                          <div key={hour} className="text-xs text-muted-foreground text-right pr-2 h-16 flex items-start">
                            {hour > 12 ? `${hour - 12}:00 PM` : hour === 12 ? '12:00 PM' : `${hour}:00 AM`}
                          </div>
                        ))}
                      </div>
                      
                      {/* Week days columns */}
                      {weekDays.map((day, dayIndex) => {
                        const date = weekDates[dayIndex];
                        const dateNumber = date.getDate();
                        const isToday = date.toDateString() === today.toDateString();
                        
                        return (
                          <div key={day} className="flex flex-col">
                            <div className={`text-center pb-2 mb-2 border-b h-14 flex flex-col justify-center ${isToday ? 'font-bold text-primary' : 'text-muted-foreground'}`}>
                              <div className="text-xs mb-1">{day}</div>
                              <div className={`text-lg ${isToday ? 'bg-primary text-primary-foreground rounded-full w-8 h-8 flex items-center justify-center mx-auto' : ''}`}>
                                {dateNumber}
                              </div>
                            </div>
                          
                          {/* Time slots */}
                          <div className="flex-1 relative">
                            <div className="absolute inset-0 flex flex-col">
                              {Array.from({ length: 15 }).map((_, i) => (
                                <div key={i} className="h-16 border-t border-muted/30"></div>
                              ))}
                            </div>
                            
                            {/* Appointments - only show on current day for demo */}
                            {isToday && mockAppointments.map((apt, idx) => {
                              const startOffset = ((apt.startHour - 8) * 60) / (15 * 60) * 100; // Calculate percentage from 8 AM start (15 hours total)
                              const height = (apt.durationMinutes / (15 * 60)) * 100; // Calculate height as percentage of 15-hour range
                              
                              // Calculate end time
                              const endHour = apt.startHour + Math.floor(apt.durationMinutes / 60);
                              const endMinutes = apt.durationMinutes % 60;
                              const endTime = `${endHour > 12 ? endHour - 12 : endHour}:${endMinutes.toString().padStart(2, '0')} ${endHour >= 12 ? 'PM' : 'AM'}`;
                              
                              return (
                                <div
                                  key={apt.id}
                                  className={`absolute ${apt.color} text-white rounded-lg p-2 left-0 right-0 mx-1 cursor-pointer hover:shadow-lg transition-shadow`}
                                  style={{
                                    top: `${startOffset}%`,
                                    height: `${height}%`,
                                    minHeight: '50px'
                                  }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    if (apt.zoomLink) {
                                      window.open(apt.zoomLink, '_blank', 'noopener,noreferrer');
                                    }
                                  }}
                                >
                                  <p className="text-xs font-semibold line-clamp-1">{apt.client}</p>
                                  <p className="text-xs opacity-90">{apt.time} - {endTime}</p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                      })}
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
    </SidebarProvider>
  )
}
