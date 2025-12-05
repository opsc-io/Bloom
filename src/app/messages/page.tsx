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
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Search, Send, MessageSquarePlus, Smile } from "lucide-react";

import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { io, Socket } from "socket.io-client";

export const dynamic = "force-dynamic";

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
  reactions?: { emoji: string; count: number }[];
};

function MessagesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const messageUserId = searchParams.get("message");
  const { data: session, isPending } = useSession();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [showEmojiPicker, setShowEmojiPicker] = useState<string | null>(null);
  const [typingMap, setTypingMap] = useState<Record<string, { userId: string; name?: string | null; expiresAt: number }>>({});
  const socketRef = React.useRef<Socket | null>(null);
  const lastTypingSentRef = React.useRef<number>(0);

  const commonEmojis = ["👍", "❤️", "😊", "😂", "🎉", "👏"];

  const handleReaction = (messageId: string, emoji: string) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) => {
        if (msg.id === messageId) {
          const reactions = msg.reactions || [];
          const existingReaction = reactions.find((r) => r.emoji === emoji);
          if (existingReaction) {
            return {
              ...msg,
              reactions: reactions.map((r) =>
                r.emoji === emoji ? { ...r, count: r.count + 1 } : r
              ),
            };
          }
          return {
            ...msg,
            reactions: [...reactions, { emoji, count: 1 }],
          };
        }
        return msg;
      })
    );
    setShowEmojiPicker(null);
  };

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
          if (messageUserId && convIds.includes(messageUserId)) {
            setActiveConversation(messageUserId);
          } else if (messageUserId) {
            // allow starting a new conversation with a user id from the query param
            setActiveConversation(messageUserId);
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
  }, [isPending, session, messageUserId, activeConversation]);

  // Join active conversation room for typing updates when it changes
  useEffect(() => {
    if (!activeConversation || !socketRef.current) return;
    socketRef.current.emit("join", { conversationId: activeConversation });
  }, [activeConversation]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    const url = process.env.NEXT_PUBLIC_SOCKET_URL || "http://localhost:4000";
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
  const isTyping =
    typingEntry &&
    typingEntry.userId !== (session?.user as { id?: string })?.id &&
    typingEntry.expiresAt > Date.now();

  if (isPending) return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user) return <p className="text-center mt-8 text-white">Redirecting...</p>;

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

        <div className="flex flex-1 flex-col p-4 h-[calc(100vh-4rem)] overflow-hidden">
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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-0 flex-1 min-h-0">
              {/* Conversations List */}
              <div className="md:col-span-1 border-r p-4 overflow-hidden flex flex-col h-full">
                <div className="relative mb-4 flex-shrink-0">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search conversations..."
                    value={conversationSearch}
                    onChange={(e) => setConversationSearch(e.target.value)}
                    className="pl-9 bg-muted/50"
                  />
                </div>
                <div className="space-y-1 overflow-y-auto flex-1">
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
              <div className="md:col-span-2 flex flex-col min-h-0">
                {activeConv ? (
                  <div className="flex flex-col h-full p-4">
                    {/* Chat Header */}
                    <div className="flex items-center gap-3 pb-4 border-b mb-4 flex-shrink-0">
                      <Avatar className="h-10 w-10">
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

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto pr-2 min-h-0">
                      <div className="space-y-4">
                        {messages.map((msg) => (
                          <div
                            key={msg.id}
                            className={`flex gap-3 ${msg.isMe ? "justify-end" : ""} group`}
                          >
                            {!msg.isMe && (
                              <Avatar className="h-8 w-8">
                                <AvatarFallback className={msg.avatarColor}>
                                  {msg.avatar}
                                </AvatarFallback>
                              </Avatar>
                            )}
                            <div
                              className={`flex flex-col gap-1 max-w-[70%] relative ${
                                msg.isMe ? "items-end" : ""
                              }`}
                            >
                              <div className="relative">
                                <div
                                  className={`rounded-lg px-4 py-2 ${
                                    msg.isMe
                                      ? "bg-primary text-primary-foreground"
                                      : "bg-muted"
                                  }`}
                                >
                                  <p className="text-sm">{msg.message}</p>
                                </div>

                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className={`absolute -bottom-2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity bg-background border shadow-sm ${
                                    msg.isMe ? "right-0" : "left-0"
                                  }`}
                                  onClick={() =>
                                    setShowEmojiPicker(showEmojiPicker === msg.id ? null : msg.id)
                                  }
                                >
                                  <Smile className="h-3 w-3" />
                                </Button>

                                {showEmojiPicker === msg.id && (
                                  <div
                                    className={`absolute bottom-8 bg-background border rounded-lg shadow-lg p-2 flex gap-1 z-10 ${
                                      msg.isMe ? "right-0" : "left-0"
                                    }`}
                                  >
                                    {commonEmojis.map((emoji) => (
                                      <button
                                        key={emoji}
                                        onClick={async () => {
                                          await fetch("/api/messages", {
                                            method: "PUT",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ messageId: msg.id, emoji }),
                                          }).catch(() => {});

                                          setMessages((prev) =>
                                            prev.map((m) => {
                                              if (m.id !== msg.id) return m;
                                              const reactions = m.reactions ?? [];
                                              const existing = reactions.find((r) => r.emoji === emoji);
                                              if (existing) {
                                                const updated = reactions
                                                  .map((r) =>
                                                    r.emoji === emoji
                                                      ? { ...r, count: r.count === 1 ? 0 : r.count + 1 }
                                                      : r
                                                  )
                                                  .filter((r) => r.count > 0);
                                                return { ...m, reactions: updated };
                                              }
                                              return { ...m, reactions: [...reactions, { emoji, count: 1 }] };
                                            })
                                          );
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
                                        await fetch("/api/messages", {
                                          method: "PUT",
                                          headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ messageId: msg.id, emoji: reaction.emoji }),
                                        }).catch(() => {});
                                        setMessages((prev) =>
                                          prev.map((m) => {
                                            if (m.id !== msg.id) return m;
                                            const reactions = m.reactions ?? [];
                                            const updated = reactions
                                              .map((r) =>
                                                r.emoji === reaction.emoji
                                                  ? { ...r, count: Math.max(0, r.count - 1) }
                                                  : r
                                              )
                                              .filter((r) => r.count > 0);
                                            return { ...m, reactions: updated };
                                          })
                                        );
                                      }}
                                    >
                                      <span>{reaction.emoji}</span>
                                      <span className="text-muted-foreground">{reaction.count}</span>
                                    </div>
                                  ))}
                                </div>
                              )}

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
                    <div className="flex gap-2 pt-4 border-t flex-shrink-0">
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
                  </div>
                ) : (
                  <div className="flex flex-col h-full relative p-4">
                    <div className="flex-1 flex items-center justify-center text-muted-foreground min-h-0">
                      <p>Select a conversation to start messaging</p>
                    </div>

                    {/* Message Input Bar - Always visible at bottom */}
                    <div className="flex gap-2 pt-4 border-t flex-shrink-0">
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

export default function MessagesPage() {
  return (
    <Suspense fallback={<p className="p-4">Loading messages...</p>}>
      <MessagesContent />
    </Suspense>
  );
}
