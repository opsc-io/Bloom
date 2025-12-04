import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { auth } from "@/lib/auth";

const colorPool = ["bg-blue-500", "bg-purple-500", "bg-green-500", "bg-amber-500", "bg-pink-500", "bg-indigo-500"];

const pickColor = (id: string) => {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash << 5) - hash + id.charCodeAt(i);
  return colorPool[Math.abs(hash) % colorPool.length];
};

const initials = (firstname?: string | null, lastname?: string | null) => {
  const first = firstname?.trim()?.[0] ?? "";
  const last = lastname?.trim()?.[0] ?? "";
  const combined = `${first}${last}`.toUpperCase();
  return combined || "U";
};

export async function GET(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userId = session.user.id;

  // Dummy conversation data for testing
  const dummyConversations = [
    {
      id: "conv-1",
      name: "Dr. Sarah Mitchell",
      avatar: "SM",
      avatarColor: "bg-blue-500",
      lastMessage: "How have you been feeling this week?",
      time: "2:30 PM",
      unread: 2,
      active: true,
    },
    {
      id: "conv-2",
      name: "Michael Chen",
      avatar: "MC",
      avatarColor: "bg-purple-500",
      lastMessage: "Thanks for the session yesterday!",
      time: "11:45 AM",
      unread: 0,
      active: false,
    },
    {
      id: "conv-3",
      name: "Emily Rodriguez",
      avatar: "ER",
      avatarColor: "bg-green-500",
      lastMessage: "Can we reschedule our appointment?",
      time: "Yesterday",
      unread: 1,
      active: false,
    },
  ];

  const dummyMessages = [
    {
      id: "msg-1",
      sender: "Dr. Sarah Mitchell",
      message: "Hi! How have you been feeling this week?",
      time: "2:25 PM",
      isMe: false,
      avatar: "SM",
      avatarColor: "bg-blue-500",
    },
    {
      id: "msg-2",
      sender: "You",
      message: "I've been doing better, thank you for asking.",
      time: "2:27 PM",
      isMe: true,
      avatar: "Y",
      avatarColor: "bg-blue-500",
    },
    {
      id: "msg-3",
      sender: "Dr. Sarah Mitchell",
      message: "That's great to hear! Have you been practicing the breathing exercises we discussed?",
      time: "2:28 PM",
      isMe: false,
      avatar: "SM",
      avatarColor: "bg-blue-500",
    },
    {
      id: "msg-4",
      sender: "You",
      message: "Yes, I've been doing them every morning and it really helps with my anxiety.",
      time: "2:29 PM",
      isMe: true,
      avatar: "Y",
      avatarColor: "bg-blue-500",
    },
    {
      id: "msg-5",
      sender: "Dr. Sarah Mitchell",
      message: "Excellent! Keep up the good work. Let's schedule our next session.",
      time: "2:30 PM",
      isMe: false,
      avatar: "SM",
      avatarColor: "bg-blue-500",
    },
  ];

  // TODO: Replace with actual database queries
  // const conversations = await prisma.conversation.findMany({
  //   where: { participants: { some: { userId } } },
  //   include: {
  //     participants: { include: { user: true } },
  //     messages: { orderBy: { createdAt: "desc" }, take: 1, include: { sender: true } },
  //   },
  //   orderBy: [{ lastMessageAt: "desc" }, { createdAt: "desc" }],
  //   take: 10,
  // });

  return NextResponse.json({
    conversations: dummyConversations,
    messages: dummyMessages,
  });
}
