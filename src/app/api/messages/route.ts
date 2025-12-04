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

  const conversations = await prisma.conversation.findMany({
    where: { participants: { some: { userId } } },
    include: {
      participants: { include: { user: true } },
      messages: { orderBy: { createdAt: "desc" }, take: 1, include: { sender: true } },
    },
    orderBy: [{ lastMessageAt: "desc" }, { createdAt: "desc" }],
    take: 10,
  });

  const formattedConversations = conversations.map((conv, index) => {
    const otherParticipant = conv.participants.find((p) => p.userId !== userId)?.user ?? conv.participants[0]?.user;
    const latestMessage = conv.messages[0];
    const displayName =
      otherParticipant
        ? `${otherParticipant.firstname} ${otherParticipant.lastname}`.trim() || otherParticipant.email
        : "Conversation";

    const formattedTime = latestMessage?.createdAt
      ? new Date(latestMessage.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : "";

    return {
      id: conv.id,
      name: displayName,
      avatar: otherParticipant ? initials(otherParticipant.firstname, otherParticipant.lastname) : "C",
      avatarColor: otherParticipant ? pickColor(otherParticipant.id) : "bg-muted",
      lastMessage: latestMessage?.body ?? "No messages yet",
      time: formattedTime,
      unread: 0,
      active: index === 0,
    };
  });

  const activeConversation = formattedConversations.find((conv) => conv.active);

  const messages = activeConversation
    ? await prisma.message.findMany({
      where: { conversationId: activeConversation.id },
      include: { sender: true },
      orderBy: { createdAt: "asc" },
      take: 50,
    })
    : [];

  const formattedMessages = messages.map((msg) => {
    const messageTime = new Date(msg.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    return {
      id: msg.id,
      sender: `${msg.sender.firstname} ${msg.sender.lastname}`.trim() || msg.sender.email,
      message: msg.body ?? "",
      time: messageTime,
      isMe: msg.senderId === userId,
      avatar: initials(msg.sender.firstname, msg.sender.lastname),
      avatarColor: pickColor(msg.sender.id),
    };
  });

  return NextResponse.json({
    conversations: formattedConversations,
    messages: formattedMessages,
  });
}
