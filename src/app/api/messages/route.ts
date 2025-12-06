import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { auth } from "@/lib/auth";
import { UserRole } from "@/generated/prisma/client";
import redis, {
  getCachedOrFetch,
  invalidateConversation,
  CACHE_KEYS,
  CACHE_TTL,
} from "@/lib/redis";

const colorPool = ["bg-blue-500", "bg-purple-500", "bg-green-500", "bg-amber-500", "bg-pink-500", "bg-indigo-500"];

const pickColor = (id: string) => {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash << 5) - hash + id.charCodeAt(i);
  return colorPool[Math.abs(hash) % colorPool.length];
};

const initials = (firstname?: string | null, lastname?: string | null, fallback?: string) => {
  const first = firstname?.trim()?.[0] ?? "";
  const last = lastname?.trim()?.[0] ?? "";
  const combined = `${first}${last}`.toUpperCase();
  return combined || fallback?.[0]?.toUpperCase() || "U";
};

const formatTime = (date: Date) =>
  date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });

export async function GET(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userId = session.user.id;
  const searchParams = new URL(req.url).searchParams;
  const requestedConversationId = searchParams.get("conversationId");

  // Fetch conversations with Redis caching
  const conversations = await getCachedOrFetch(
    CACHE_KEYS.userConversations(userId),
    CACHE_TTL.conversations,
    async () => {
      return prisma.conversation.findMany({
        where: { participants: { some: { userId } } },
        include: {
          participants: { include: { user: true } },
          messages: {
            include: { sender: true, reactions: true },
            orderBy: { createdAt: "desc" },
            take: 1,
          },
        },
        orderBy: [{ lastMessageAt: "desc" }, { createdAt: "desc" }],
        take: 10,
      });
    }
  );

  const activeConversationId =
    requestedConversationId && conversations.some((c) => c.id === requestedConversationId)
      ? requestedConversationId
      : conversations[0]?.id;

  const mappedConversations = conversations.map((conv) => {
    const lastMsg = conv.messages[0];
    const otherParticipant = conv.participants.find((p) => p.userId !== userId)?.user ?? conv.participants[0]?.user;
    const name =
      otherParticipant?.firstname || otherParticipant?.lastname
        ? `${otherParticipant?.firstname ?? ""} ${otherParticipant?.lastname ?? ""}`.trim()
        : otherParticipant?.email ?? "Conversation";
    const avatarInitials = initials(otherParticipant?.firstname, otherParticipant?.lastname, otherParticipant?.email);
    return {
      id: conv.id,
      name,
      avatar: avatarInitials,
      avatarColor: pickColor(conv.id),
      lastMessage: lastMsg?.body ?? "",
      time: lastMsg ? formatTime(new Date(lastMsg.createdAt)) : "",
      unread: 0,
      active: conv.id === activeConversationId,
    };
  });

  let mappedMessages: Array<{
    id: string;
    sender: string;
    message: string;
    time: string;
    isMe: boolean;
    avatar: string;
    avatarColor: string;
    reactions: { emoji: string; count: number }[];
  }> = [];

  if (activeConversationId) {
    // Fetch messages with Redis caching
    const messages = await getCachedOrFetch(
      CACHE_KEYS.conversationMessages(activeConversationId),
      CACHE_TTL.messages,
      async () => {
        return prisma.message.findMany({
          where: { conversationId: activeConversationId },
          include: { sender: true, reactions: true },
          orderBy: { createdAt: "asc" },
          take: 30,
        });
      }
    );

    mappedMessages = messages.map((msg) => {
      const reactionCounts = Object.values(
        (msg.reactions ?? []).reduce<Record<string, { emoji: string; count: number }>>((acc, r) => {
          acc[r.emoji] = acc[r.emoji] ? { emoji: r.emoji, count: acc[r.emoji].count + 1 } : { emoji: r.emoji, count: 1 };
          return acc;
        }, {})
      );

      return {
        id: msg.id,
        sender:
          msg.sender.firstname || msg.sender.lastname
            ? `${msg.sender.firstname} ${msg.sender.lastname}`.trim()
            : msg.sender.email,
        message: msg.body ?? "",
        time: formatTime(new Date(msg.createdAt)),
        isMe: msg.senderId === userId,
        avatar: initials(msg.sender.firstname, msg.sender.lastname, msg.sender.email),
        avatarColor: pickColor(msg.senderId),
        reactions: reactionCounts,
      };
    });
  }

  return NextResponse.json({
    conversations: mappedConversations,
    messages: mappedMessages,
  });
}

export async function POST(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userId = session.user.id;
  const body = await req.json();
  const { conversationId, recipientId, message, startOnly } = body as {
    conversationId?: string;
    recipientId?: string;
    message?: string;
    startOnly?: boolean;
  };

  const messageText = typeof message === "string" ? message.trim() : "";
  const initiateOnly = Boolean(startOnly);

  if (!initiateOnly && !messageText) {
    return NextResponse.json({ error: "Message is required" }, { status: 400 });
  }

  if (!conversationId && !recipientId) {
    return NextResponse.json({ error: "conversationId or recipientId is required" }, { status: 400 });
  }

  let targetConversationId = conversationId;

  // Validate or create conversation
  if (targetConversationId) {
    const participant = await prisma.conversationParticipant.findFirst({
      where: { conversationId: targetConversationId, userId },
      select: { id: true },
    });
    if (!participant) {
      return NextResponse.json({ error: "Not a participant in this conversation" }, { status: 403 });
    }
  } else if (recipientId) {
    // Ensure the recipient exists
    const recipient = await prisma.user.findUnique({
      where: { id: recipientId },
      select: { id: true, firstname: true, lastname: true, email: true, role: true },
    });

    if (!recipient) {
      return NextResponse.json({ error: "Recipient not found" }, { status: 404 });
    }

    // Find existing conversation between the two participants
    const existingConversation = await prisma.conversation.findFirst({
      where: {
        participants: {
          some: { userId },
        },
        AND: {
          participants: {
            some: { userId: recipientId },
          },
        },
      },
      select: { id: true },
    });

    if (existingConversation) {
      targetConversationId = existingConversation.id;
    } else {
      // Create a new conversation
      const myRole = (session.user as { role?: string }).role ?? "UNSET";
      const recipientRole = recipient.role ?? "UNSET";
      const normalizeRole = (role: string | null | undefined) =>
        role && role in UserRole ? (role as keyof typeof UserRole) : "UNSET";

      const newConversation = await prisma.conversation.create({
        data: {
          lastMessageAt: new Date(),
          participants: {
            create: [
              { userId, role: UserRole[normalizeRole(myRole)] },
              { userId: recipient.id, role: UserRole[normalizeRole(recipientRole)] },
            ],
          },
        },
      });
      targetConversationId = newConversation.id;
    }
  }

  if (!targetConversationId) {
    return NextResponse.json({ error: "Unable to resolve conversation" }, { status: 400 });
  }

  // If we just want to ensure a conversation exists, update freshness and return it
  if (initiateOnly) {
    await prisma.conversation.update({
      where: { id: targetConversationId },
      data: { lastMessageAt: new Date() },
    });

    const conversation = await prisma.conversation.findUnique({
      where: { id: targetConversationId },
      include: {
        participants: { include: { user: true } },
        messages: { include: { sender: true }, orderBy: { createdAt: "desc" }, take: 1 },
      },
    });

    if (!conversation) {
      return NextResponse.json({ error: "Conversation not found after creating" }, { status: 404 });
    }

    const otherParticipant =
      conversation.participants.find((p) => p.userId !== userId)?.user ?? conversation.participants[0]?.user;
    const name =
      otherParticipant?.firstname || otherParticipant?.lastname
        ? `${otherParticipant?.firstname ?? ""} ${otherParticipant?.lastname ?? ""}`.trim()
        : otherParticipant?.email ?? "Conversation";
    const avatarInitials = initials(otherParticipant?.firstname, otherParticipant?.lastname, otherParticipant?.email);
    const lastMsg = conversation.messages[0];

    const mappedConversation = {
      id: conversation.id,
      name,
      avatar: avatarInitials,
      avatarColor: pickColor(conversation.id),
      lastMessage: lastMsg?.body ?? "",
      time: lastMsg ? formatTime(new Date(lastMsg.createdAt)) : "",
      unread: 0,
      active: true,
    };

    return NextResponse.json({ conversation: mappedConversation });
  }

  const createdMessage = await prisma.message.create({
    data: {
      conversationId: targetConversationId,
      senderId: userId,
      body: messageText,
    },
  });

  // Update last message timestamp
  await prisma.conversation.update({
    where: { id: targetConversationId },
    data: { lastMessageAt: createdMessage.createdAt },
  });

  // Get participant IDs for cache invalidation
  const participants = await prisma.conversationParticipant.findMany({
    where: { conversationId: targetConversationId },
    select: { userId: true },
  });
  const participantIds = participants.map((p) => p.userId);

  // Invalidate caches for all participants
  await invalidateConversation(targetConversationId, participantIds);

  const conversation = await prisma.conversation.findUnique({
    where: { id: targetConversationId },
    include: {
      participants: { include: { user: true } },
      messages: {
        include: { sender: true },
        orderBy: { createdAt: "desc" },
        take: 1,
      },
    },
  });

  if (!conversation) {
    return NextResponse.json({ error: "Conversation not found after sending" }, { status: 404 });
  }

  const otherParticipant =
    conversation.participants.find((p) => p.userId !== userId)?.user ?? conversation.participants[0]?.user;
  const name =
    otherParticipant?.firstname || otherParticipant?.lastname
      ? `${otherParticipant?.firstname ?? ""} ${otherParticipant?.lastname ?? ""}`.trim()
      : otherParticipant?.email ?? "Conversation";
  const avatarInitials = initials(otherParticipant?.firstname, otherParticipant?.lastname, otherParticipant?.email);
  const lastMsg = conversation.messages[0];

  const mappedConversation = {
    id: conversation.id,
    name,
    avatar: avatarInitials,
    avatarColor: pickColor(conversation.id),
    lastMessage: lastMsg?.body ?? "",
    time: lastMsg ? formatTime(new Date(lastMsg.createdAt)) : "",
    unread: 0,
    active: true,
  };

  const mappedMessage = {
    id: createdMessage.id,
    sender:
      session.user.firstname || session.user.lastname
        ? `${session.user.firstname ?? ""} ${session.user.lastname ?? ""}`.trim()
        : session.user.email,
    message: messageText,
    time: formatTime(new Date(createdMessage.createdAt)),
    isMe: true,
    avatar: initials(session.user.firstname, session.user.lastname, session.user.email),
    avatarColor: pickColor(userId),
    senderId: userId,
  };

  // Publish message to Redis for real-time delivery via Socket.io
  try {
    await redis.publish(
      `message:${targetConversationId}`,
      JSON.stringify({
        conversationId: targetConversationId,
        message: mappedMessage,
      })
    );
  } catch {
    // Ignore Redis publish errors - message is already saved
  }

  return NextResponse.json({
    conversation: mappedConversation,
    message: mappedMessage,
  });
}

export async function PUT(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userId = session.user.id;
  const body = await req.json();
  const { messageId, emoji } = body as { messageId?: string; emoji?: string };

  if (!messageId || !emoji) {
    return NextResponse.json({ error: "messageId and emoji are required" }, { status: 400 });
  }

  const message = await prisma.message.findUnique({
    where: { id: messageId },
    select: { conversationId: true },
  });
  if (!message) {
    return NextResponse.json({ error: "Message not found" }, { status: 404 });
  }

  const participant = await prisma.conversationParticipant.findFirst({
    where: { conversationId: message.conversationId, userId },
    select: { id: true },
  });
  if (!participant) {
    return NextResponse.json({ error: "Not authorized for this conversation" }, { status: 403 });
  }

  const existing = await prisma.messageReaction.findUnique({
    where: {
      messageId_userId_emoji: {
        messageId,
        userId,
        emoji,
      },
    },
  });

  if (existing) {
    await prisma.messageReaction.delete({
      where: { id: existing.id },
    });
  } else {
    await prisma.messageReaction.create({
      data: {
        messageId,
        userId,
        emoji,
      },
    });
  }

  // Invalidate message cache for this conversation
  const participants = await prisma.conversationParticipant.findMany({
    where: { conversationId: message.conversationId },
    select: { userId: true },
  });
  await invalidateConversation(
    message.conversationId,
    participants.map((p) => p.userId)
  );

  const reactions = await prisma.messageReaction.findMany({
    where: { messageId },
    select: { emoji: true },
  });

  const aggregated = Object.values(
    reactions.reduce<Record<string, { emoji: string; count: number }>>((acc, r) => {
      acc[r.emoji] = acc[r.emoji] ? { emoji: r.emoji, count: acc[r.emoji].count + 1 } : { emoji: r.emoji, count: 1 };
      return acc;
    }, {})
  );

  return NextResponse.json({ reactions: aggregated });
}
