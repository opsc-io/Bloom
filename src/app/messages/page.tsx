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
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Search, Send, MessageSquarePlus } from "lucide-react";

import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState } from "react";

type Conversation = {
  id: string;
  name: string;
  avatar: string;
  avatarColor: string;
  lastMessage: string;
  time: string;
  unread: number;
  active: boolean;
};

type Message = {
  id: string;
  sender: string;
  message: string;
  time: string;
  isMe: boolean;
  avatar: string;
  avatarColor: string;
};

export default function MessagesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const messageUserId = searchParams.get("message");
  const { data: session, isPending } = useSession();
  
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;

    const loadMessages = async () => {
      try {
        const res = await fetch(
          activeConversation ? `/api/messages?conversationId=${activeConversation}` : "/api/messages"
        );
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setConversations(data.conversations ?? []);
        setMessages(data.messages ?? []);

        // Determine active conversation: query param, existing active, or first available
        const convIds = (data.conversations ?? []).map((c: Conversation) => c.id);
        if (activeConversation && !convIds.includes(activeConversation)) {
          setActiveConversation(data.conversations?.[0]?.id ?? null);
        } else if (!activeConversation) {
          if (messageUserId && convIds.includes(messageUserId)) {
            setActiveConversation(messageUserId);
          } else if (messageUserId) {
            // allow starting a new conversation with a user id from the query param
            setActiveConversation(messageUserId);
          } else if (data.conversations?.length > 0) {
            setActiveConversation(data.conversations[0].id);
          }
        }
      } catch {
        // Handle error silently
      }
    };

    loadMessages();
    return () => {
      cancelled = true;
    };
  }, [isPending, session, messageUserId, activeConversation]);

  const handleSendMessage = async () => {
    if (!newMessage.trim() || !activeConversation) return;

    const payload: Record<string, string> = { message: newMessage.trim() };
    const conversationExists = conversations.some((c) => c.id === activeConversation);
    if (conversationExists) {
      payload.conversationId = activeConversation;
    } else {
      payload.recipientId = activeConversation;
    }

    try {
      const res = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        return;
      }

      const data = await res.json() as { conversation?: Conversation; message?: Message };
      if (data.conversation) {
        setConversations((prev) => {
          const filtered = prev.filter((c) => c.id !== data.conversation!.id).map((c) => ({ ...c, active: false }));
          return [{ ...data.conversation!, active: true }, ...filtered];
        });
        setActiveConversation(data.conversation.id);
      }
      if (data.message) {
        setMessages((prev) => [...prev, data.message!]);
      }
      setNewMessage("");
    } catch {
      // ignore errors for now
    }
  };

  const filteredConversations = conversations.filter((conv) =>
    conv.name.toLowerCase().includes(conversationSearch.toLowerCase())
  );

  const activeConv = conversations.find((c) => c.id === activeConversation);

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;

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
                <BreadcrumbPage>Messages</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col">
          <div className="flex-1 bg-card/50 backdrop-blur-sm border border-border/50 rounded-lg">
            <div className="p-6 border-b">
              <h2 className="text-xl font-semibold">Messages</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Communicate with your patients and therapists
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-0 h-[calc(100vh-240px)]">
              {/* Conversations List */}
              <div className="md:col-span-1 border-r p-4">
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search conversations..."
                    value={conversationSearch}
                    onChange={(e) => setConversationSearch(e.target.value)}
                    className="pl-9 bg-muted/50"
                  />
                </div>
                <div className="space-y-1 overflow-y-auto h-[calc(100vh-340px)]">
                      {filteredConversations.map((conversation) => (
                        <div
                          key={conversation.id}
                          onClick={() => setActiveConversation(conversation.id)}
                          className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                            activeConversation === conversation.id
                              ? "bg-muted"
                              : "hover:bg-muted/50"
                          }`}
                        >
                          <Avatar className="h-10 w-10">
                            <AvatarFallback className={conversation.avatarColor}>
                              {conversation.avatar}
                            </AvatarFallback>
                          </Avatar>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-1">
                              <p className="text-sm font-medium truncate">
                                {conversation.name}
                              </p>
                              <span className="text-xs text-muted-foreground">
                                {conversation.time}
                              </span>
                            </div>
                            <p className="text-xs text-muted-foreground truncate">
                              {conversation.lastMessage}
                            </p>
                          </div>
                          {conversation.unread > 0 && (
                            <div className="flex items-center justify-center h-5 w-5 rounded-full bg-primary text-primary-foreground text-xs">
                              {conversation.unread}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Messages Area */}
                  <div className="md:col-span-2 flex flex-col">
                    {activeConv ? (
                      <div className="flex flex-col h-full p-4">
                        {/* Chat Header */}
                        <div className="flex items-center gap-3 pb-4 border-b mb-4">
                          <Avatar className="h-10 w-10">
                            <AvatarFallback className={activeConv.avatarColor}>
                              {activeConv.avatar}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-medium">{activeConv.name}</p>
                            <p className="text-xs text-muted-foreground">Active now</p>
                          </div>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                          {messages.map((msg) => (
                            <div
                              key={msg.id}
                              className={`flex gap-3 ${msg.isMe ? "justify-end" : ""}`}
                            >
                              {!msg.isMe && (
                                <Avatar className="h-8 w-8">
                                  <AvatarFallback className={msg.avatarColor}>
                                    {msg.avatar}
                                  </AvatarFallback>
                                </Avatar>
                              )}
                              <div
                                className={`flex flex-col gap-1 max-w-[70%] ${
                                  msg.isMe ? "items-end" : ""
                                }`}
                              >
                                <div
                                  className={`rounded-lg px-4 py-2 ${
                                    msg.isMe
                                      ? "bg-primary text-primary-foreground"
                                      : "bg-muted"
                                  }`}
                                >
                                  <p className="text-sm">{msg.message}</p>
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {msg.time}
                                </span>
                              </div>
                              {msg.isMe && (
                                <Avatar className="h-8 w-8">
                                  <AvatarFallback className={msg.avatarColor}>
                                    {msg.avatar}
                                  </AvatarFallback>
                                </Avatar>
                              )}
                            </div>
                          ))}
                        </div>

                        {/* Message Input */}
                        <div className="flex gap-2 pt-4 border-t">
                          <Input
                            placeholder="Type a message..."
                            value={newMessage}
                            onChange={(e) => setNewMessage(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                handleSendMessage();
                              }
                            }}
                            className="flex-1"
                          />
                          <Button onClick={handleSendMessage} size="icon">
                            <Send className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col h-full relative p-4">
                        <div className="flex-1 flex items-center justify-center text-muted-foreground">
                          <p>Select a conversation to start messaging</p>
                        </div>
                        
                        {/* Message Input Bar - Always visible at bottom */}
                        <div className="flex gap-2 pt-4 border-t">
                          <Input 
                            placeholder="Type a message..." 
                            className="flex-1"
                            disabled
                          />
                          <Button size="icon" className="shrink-0" disabled>
                            <Send className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </SidebarInset>
        </SidebarProvider>
      );
    }
