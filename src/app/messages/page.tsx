"use client";

import React, { Suspense, useEffect, useState } from "react";
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
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Search, Send, MessageSquarePlus, Smile, BarChart3 } from "lucide-react";

import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { io, Socket } from "socket.io-client";

export const dynamic = "force-dynamic";

type Conversation = {
  id: string;
  name: string;
  avatar: string;
  avatarColor: string;
  image?: string | null;
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
  image?: string | null;
  reactions?: { emoji: string; count: number }[];
};


function MessagesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const conversationParam = searchParams.get("conversationId");
  const recipientParam = conversationParam ? null : searchParams.get("message");
  const requestedConversationId = conversationParam ?? recipientParam;
  const { data: session, isPending } = useSession();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(requestedConversationId ?? null);
  const [newMessage, setNewMessage] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [showEmojiPicker, setShowEmojiPicker] = useState<string | null>(null);
  const [typingMap, setTypingMap] = useState<Record<string, { userId: string; name?: string | null; expiresAt: number }>>({});
  const socketRef = React.useRef<Socket | null>(null);
  const lastTypingSentRef = React.useRef<number>(0);

  const commonEmojis = ["👍", "❤️", "😊", "😂", "🎉", "👏"];

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

        const convIds = (data.conversations ?? []).map((c: Conversation) => c.id);
        if (activeConversation && !convIds.includes(activeConversation)) {
          setActiveConversation(data.conversations?.[0]?.id ?? null);
        } else if (!activeConversation) {
          if (requestedConversationId && convIds.includes(requestedConversationId)) {
            setActiveConversation(requestedConversationId);
          } else if (requestedConversationId) {
            // allow starting a new conversation with a user id from the query param
            setActiveConversation(requestedConversationId);
          } else if (data.conversations?.length > 0) {
            setActiveConversation(data.conversations[0].id);
          }
        }

        // Join the first/active conversation room for typing signals
        const joinId = activeConversation ?? data.conversations?.[0]?.id;
        if (joinId && socketRef.current) {
          socketRef.current.emit("join", { conversationId: joinId });
        }
      } catch {
        // ignore to keep page usable
      }
    };

    loadMessages();
    return () => {
      cancelled = true;
    };
  }, [isPending, session, requestedConversationId, activeConversation]);

  // If user is deep-linked with a user id (not an existing conversation), create the conversation first
  useEffect(() => {
    if (isPending || !session?.user || !recipientParam) return;
    const alreadyExists = conversations.some((c) => c.id === recipientParam);
    if (alreadyExists) return;

    let cancelled = false;
    const ensureConversation = async () => {
      try {
        const res = await fetch("/api/messages", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ recipientId: recipientParam, startOnly: true }),
        });
        if (!res.ok) return;
        const data = await res.json();
        const convoId = data.conversation?.id;
        if (!convoId || cancelled) return;
        setActiveConversation(convoId);

        const refresh = await fetch(`/api/messages?conversationId=${convoId}`);
        if (!refresh.ok) return;
        const refreshed = await refresh.json();
        if (cancelled) return;
        setConversations(refreshed.conversations ?? []);
        setMessages(refreshed.messages ?? []);
      } catch {
        // keep page usable even if ensureConversation fails
      }
    };

    ensureConversation();
    return () => {
      cancelled = true;
    };
  }, [isPending, session, recipientParam, conversations]);

  // Join active conversation room for typing updates when it changes
  useEffect(() => {
    if (!activeConversation || !socketRef.current) return;
    socketRef.current.emit("join", { conversationId: activeConversation });
  }, [activeConversation]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    // Socket connects to same origin as app (via /socket.io/* path in ingress)
    const url = typeof window !== "undefined" ? window.location.origin : "http://localhost:4000";
    const socket = io(url, {
      withCredentials: true,
      auth: {
        token: typeof document !== "undefined" ? document.cookie : undefined,
      },
    });
    socketRef.current = socket;

    socket.on("typing", (payload: { conversationId: string; userId: string; name?: string | null; expiresAt: number }) => {
      setTypingMap((prev) => ({
        ...prev,
        [payload.conversationId]: payload,
      }));
    });

    // Listen for new messages from other users
    socket.on("newMessage", (payload: { conversationId: string; message: Message & { senderId: string } }) => {
      const currentUserId = (session?.user as { id?: string })?.id;
      // Only add if it's from another user (not our own message)
      if (payload.message.senderId !== currentUserId) {
        setMessages((prev) => {
          // Avoid duplicates
          if (prev.some((m) => m.id === payload.message.id)) return prev;
          return [...prev, { ...payload.message, isMe: false }];
        });
        // Update conversation list with new last message
        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === payload.conversationId
              ? { ...conv, lastMessage: payload.message.message, time: payload.message.time }
              : conv
          )
        );
      }
    });

    // Listen for reaction updates
    socket.on("reactionUpdate", (payload: { conversationId: string; messageId: string; reactions: { emoji: string; count: number }[] }) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === payload.messageId ? { ...msg, reactions: payload.reactions } : msg
        )
      );
    });

    const interval = setInterval(() => {
      const now = Date.now();
      setTypingMap((prev) => {
        const next = { ...prev };
        let changed = false;
        for (const key of Object.keys(next)) {
          if (next[key].expiresAt < now) {
            delete next[key];
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }, 1000);

    return () => {
      clearInterval(interval);
      socket.disconnect();
      socketRef.current = null;
    };
  }, [isPending, session]);

  const handleSendMessage = async (overrideRecipientId?: string) => {
    if (!newMessage.trim() && !overrideRecipientId) return;
    const target = overrideRecipientId ?? activeConversation;
    if (!target) return;

    const payload: Record<string, string> = {
      message: newMessage.trim(),
    };
    const conversationExists = conversations.some((c) => c.id === target);
    if (conversationExists) {
      payload.conversationId = target;
    } else {
      payload.recipientId = target;
    }

    try {
      const res = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) return;

      const data = (await res.json()) as { conversation?: Conversation; message?: Message };
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
      if (!overrideRecipientId) {
        setNewMessage("");
      }
      setShowEmojiPicker(null);
    } catch {
      // ignore errors for now
    }
  };

  const filteredConversations = conversations.filter((conv) =>
    conv.name.toLowerCase().includes(conversationSearch.toLowerCase())
  );

  const activeConv = conversations.find((c) => c.id === activeConversation);
  const typingEntry = activeConversation ? typingMap[activeConversation] : undefined;
  const [currentTime, setCurrentTime] = useState(() => Date.now());

  // Update current time periodically for typing indicator expiry check
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  const isTyping =
    typingEntry &&
    typingEntry.userId !== (session?.user as { id?: string })?.id &&
    typingEntry.expiresAt > currentTime;

  if (isPending) return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user) return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;
  const userRole = (user as { role?: string }).role || "UNSET";

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

        <div className="flex flex-1 flex-col p-4 h-[calc(100vh-4rem)] max-h-[calc(100vh-4rem)] min-h-[calc(100vh-4rem)] overflow-hidden">
          <div className="flex-1 bg-card/50 backdrop-blur-sm border border-border/50 rounded-lg flex flex-col overflow-hidden">
            <div className="p-6 border-b flex-shrink-0">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Messages</h2>
                  <p className="text-sm text-muted-foreground mt-1">
                    Communicate with your patients and therapists
                  </p>
                </div>
                <Button variant="outline" size="sm" className="gap-2">
                  <MessageSquarePlus className="h-4 w-4" />
                  New Message
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-0 flex-1 min-h-0 h-full max-h-full">
              {/* Conversations List */}
              <div className="md:col-span-1 border-r overflow-auto flex flex-col min-h-0">
                <div className="p-4 flex-shrink-0">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search conversations..."
                      value={conversationSearch}
                    onChange={(e) => setConversationSearch(e.target.value)}
                    className="pl-9 bg-muted/50"
                  />
                  </div>
                </div>
                <div className="p-4 space-y-1 flex-1">
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
                        {conversation.image && <AvatarImage src={conversation.image} alt={conversation.name} />}
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
              <div className="md:col-span-2 flex flex-col min-h-0 h-full max-h-full overflow-hidden">
                {activeConv ? (
                  <>
                    {/* Chat Header */}
                    <div className="flex items-center gap-3 p-4 pb-4 border-b">
                      <Avatar className="h-10 w-10">
                        {activeConv.image && <AvatarImage src={activeConv.image} alt={activeConv.name} />}
                        <AvatarFallback className={activeConv.avatarColor}>
                          {activeConv.avatar}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <p className="font-medium">{activeConv.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {isTyping ? (
                            <span className="flex items-center gap-1">
                              <span className="animate-pulse">
                                {typingEntry?.name ? `${typingEntry.name} is typing` : "Typing"}
                              </span>
                              <span className="flex gap-0.5">
                                <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></span>
                                <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></span>
                                <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></span>
                              </span>
                            </span>
                          ) : (
                            "Active now"
                          )}
                        </p>
                      </div>
                    </div>

                    {/* Messages - Scrollable area */}
                    <div className="flex-1 overflow-y-auto overflow-x-visible p-4 min-h-0 bg-muted/10 relative">
                      <div className="space-y-4 max-w-3xl mx-auto w-full">
                      {messages.map((msg, idx) => (
                        <div
                          key={msg.id}
                          className={`flex gap-3 ${msg.isMe ? "justify-end" : ""} group`}
                        >
                          {!msg.isMe && (
                            <Avatar className="h-8 w-8">
                              {msg.image && <AvatarImage src={msg.image} alt={msg.sender} />}
                              <AvatarFallback className={msg.avatarColor}>
                                {msg.avatar}
                              </AvatarFallback>
                            </Avatar>
                          )}
                          <div className={`flex flex-col gap-1 max-w-[70%] relative ${msg.isMe ? "items-end" : ""}`}>
                            <div className="relative flex items-center gap-2">
                              <div
                                className={`rounded-lg px-4 py-2 ${
                                  msg.isMe ? "bg-primary text-primary-foreground" : "bg-muted"
                                }`}
                              >
                                <p className="text-sm">{msg.message}</p>
                              </div>

                              {!msg.isMe && userRole === "THERAPIST" && (
                                <div className="relative group/insight">
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    className="h-7 w-7 p-0 border bg-background/80 hover:bg-background"
                                    aria-label="Patient insights"
                                  >
                                    <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
                                  </Button>
                                  <div className="pointer-events-none absolute left-full top-0 ml-2 hidden w-56 rounded-lg border bg-card p-3 shadow-lg group-hover/insight:block z-30">
                                    <p className="text-xs font-semibold mb-1">Patient snapshot</p>
                                    <div className="text-xs text-muted-foreground space-y-1">
                                      <p className="flex items-center gap-2">
                                        <span className="h-2.5 w-2.5 rounded-full bg-green-500"></span>
                                        Mood trend: steady ↑
                                      </p>
                                      <p className="flex items-center gap-2">
                                        <span className="h-2.5 w-2.5 rounded-full bg-green-400"></span>
                                        Sleep: improving (6.8h avg)
                                      </p>
                                      <p className="flex items-center gap-2">
                                        <span className="h-2.5 w-2.5 rounded-full bg-yellow-400"></span>
                                        Check-ins this week: 3
                                      </p>
                                      <p className="flex items-center gap-2">
                                        <span className="h-2.5 w-2.5 rounded-full bg-red-400"></span>
                                        Next step: reinforce evening routine
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              )}

                              <Button
                                size="sm"
                                variant="ghost"
                                className={`absolute -bottom-2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity bg-background border shadow-sm ${
                                  msg.isMe ? "right-0" : "left-0"
                                }`}
                                onClick={() => setShowEmojiPicker(showEmojiPicker === msg.id ? null : msg.id)}
                              >
                                <Smile className="h-3 w-3" />
                              </Button>

                              {showEmojiPicker === msg.id && (
                                <div
                                  className={`absolute ${
                                    idx === 0 ? "top-full mt-2" : "bottom-8"
                                  } bg-background border rounded-lg shadow-lg p-2 flex gap-1 z-10 ${
                                    msg.isMe ? "right-0" : "left-0"
                                  }`}
                                >
                                  {commonEmojis.map((emoji) => (
                                    <button
                                      key={emoji}
                                      onClick={async () => {
                                        try {
                                          const res = await fetch("/api/messages", {
                                            method: "PUT",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ messageId: msg.id, emoji }),
                                          });
                                          if (res.ok) {
                                            const data = await res.json();
                                            setMessages((prev) =>
                                              prev.map((m) => (m.id === msg.id ? { ...m, reactions: data.reactions ?? [] } : m))
                                            );
                                          }
                                        } catch {
                                          // Ignore errors
                                        }
                                        setShowEmojiPicker(null);
                                      }}
                                      className="hover:bg-muted rounded p-1 text-lg transition-colors"
                                    >
                                      {emoji}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>

                            {msg.reactions && msg.reactions.length > 0 && (
                              <div className="flex gap-1 flex-wrap">
                                {msg.reactions.map((reaction, idx) => (
                                  <div
                                    key={idx}
                                    className="bg-background border rounded-full px-2 py-0.5 text-xs flex items-center gap-1 cursor-pointer hover:bg-muted transition-colors"
                                    onClick={async () => {
                                      try {
                                        const res = await fetch("/api/messages", {
                                          method: "PUT",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ messageId: msg.id, emoji: reaction.emoji }),
                                        });
                                        if (res.ok) {
                                          const data = await res.json();
                                          setMessages((prev) =>
                                            prev.map((m) => (m.id === msg.id ? { ...m, reactions: data.reactions ?? [] } : m))
                                          );
                                        }
                                      } catch {
                                        // Ignore errors
                                      }
                                    }}
                                  >
                                    <span>{reaction.emoji}</span>
                                    <span className="text-muted-foreground">{reaction.count}</span>
                                  </div>
                                ))}
                              </div>
                            )}

                            <span className="text-xs text-muted-foreground">{msg.time}</span>
                          </div>
                          {msg.isMe && (
                            <Avatar className="h-8 w-8">
                              {msg.image && <AvatarImage src={msg.image} alt={msg.sender} />}
                              <AvatarFallback className={msg.avatarColor}>{msg.avatar}</AvatarFallback>
                            </Avatar>
                          )}
                        </div>
                      ))}

                      {isTyping && (
                        <div className="flex gap-3 justify-start animate-in fade-in slide-in-from-bottom-2 duration-300">
                            <Avatar className="h-8 w-8 animate-in zoom-in duration-300">
                              <AvatarFallback className="bg-blue-500">
                                {typingEntry?.name
                                  ? typingEntry.name
                                      .split(" ")
                                      .map((n) => n[0])
                                      .join("")
                                      .toUpperCase()
                                  : "T"}
                              </AvatarFallback>
                            </Avatar>
                            <div className="flex flex-col gap-1 items-start">
                              <div className="rounded-lg px-4 py-3 bg-primary/10 border border-primary/20">
                                <div className="flex gap-1.5">
                                  <span
                                    className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                                    style={{ animationDelay: "0ms", animationDuration: "1s" }}
                                  ></span>
                                  <span
                                    className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                                    style={{ animationDelay: "150ms", animationDuration: "1s" }}
                                  ></span>
                                  <span
                                    className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                                    style={{ animationDelay: "300ms", animationDuration: "1s" }}
                                  ></span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Message Input */}
                    <div className="flex gap-2 p-4 border-t flex-shrink-0">
                      <Input
                        placeholder="Type a message..."
                        value={newMessage}
                        onChange={(e) => {
                          setNewMessage(e.target.value);
                          const now = Date.now();
                          if (now - lastTypingSentRef.current > 1000 && socketRef.current && activeConversation) {
                            socketRef.current.emit("typing", { conversationId: activeConversation });
                            lastTypingSentRef.current = now;
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        className="flex-1"
                      />
                      <Button
                        onClick={() => {
                          handleSendMessage();
                        }}
                        size="icon"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex-1 flex items-center justify-center text-muted-foreground p-4">
                      <p>Select a conversation to start messaging</p>
                    </div>

                    {/* Message Input Bar - Always visible at bottom */}
                    <div className="flex gap-2 p-4 border-t flex-shrink-0">
                      <Input
                        placeholder="Type a message..."
                        className="flex-1"
                        disabled
                      />
                      <Button size="icon" className="shrink-0" disabled>
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default function MessagesPage() {
  return (
    <Suspense fallback={<p className="p-4">Loading messages...</p>}>
      <MessagesContent />
    </Suspense>
  );
}
