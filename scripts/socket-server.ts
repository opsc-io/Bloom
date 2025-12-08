import "dotenv/config";
import { createServer } from "http";
import { Server, Socket } from "socket.io";
import Redis from "ioredis";
import prisma from "../src/lib/prisma";
import { auth } from "../src/lib/auth";

interface AuthenticatedSocket extends Socket {
  userId: string;
  userName?: string;
}

const SOCKET_PORT = Number(process.env.SOCKET_PORT || 4000);
const SOCKET_CORS = process.env.SOCKET_CORS_ORIGIN
  ? process.env.SOCKET_CORS_ORIGIN.split(",")
  : ["http://localhost:3000", "http://127.0.0.1:3000"];
const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";

const pub = new Redis(REDIS_URL);
const sub = new Redis(REDIS_URL);

// Log Redis connection status
pub.on("connect", () => console.log("[Redis] Publisher connected to:", REDIS_URL));
pub.on("ready", () => console.log("[Redis] Publisher ready"));
pub.on("error", (err) => console.error("[Redis] Publisher error:", err));
pub.on("close", () => console.log("[Redis] Publisher connection closed"));

sub.on("connect", () => console.log("[Redis] Subscriber connected to:", REDIS_URL));
sub.on("ready", () => console.log("[Redis] Subscriber ready"));
sub.on("error", (err) => console.error("[Redis] Subscriber error:", err));
sub.on("close", () => console.log("[Redis] Subscriber connection closed"));

// Log when psubscribe succeeds
sub.on("psubscribe", (pattern, count) => {
  console.log(`[Redis] Subscribed to pattern: ${pattern}, total subscriptions: ${count}`);
});

type TypingPayload = {
  conversationId: string;
  userId: string;
  name?: string | null;
  expiresAt: number;
};

type MessagePayload = {
  conversationId: string;
  message: {
    id: string;
    sender: string;
    message: string;
    time: string;
    isMe: boolean;
    avatar: string;
    avatarColor: string;
    senderId: string;
  };
};

type ReactionPayload = {
  conversationId: string;
  messageId: string;
  reactions: { emoji: string; count: number }[];
};

async function getSessionFromSocket(socket: Socket) {
  const headers = new Headers();
  const cookie = socket.handshake?.headers?.cookie;
  if (cookie) headers.set("cookie", cookie);
  const bearer = (socket.handshake?.auth as { token?: string })?.token;
  if (bearer) headers.set("authorization", `Bearer ${bearer}`);
  try {
    return await auth.api.getSession({ headers });
  } catch {
    return null;
  }
}

async function userConversationIds(userId: string) {
  const parts = await prisma.conversationParticipant.findMany({
    where: { userId },
    select: { conversationId: true },
  });
  return parts.map((p) => p.conversationId);
}

async function userIsParticipant(userId: string, conversationId: string) {
  const participant = await prisma.conversationParticipant.findFirst({
    where: { userId, conversationId },
    select: { id: true },
  });
  return Boolean(participant);
}

async function main() {
  const httpServer = createServer((req, res) => {
    // Health check endpoint
    if (req.url === "/health") {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("ok");
      return;
    }
  });

  const io = new Server(httpServer, {
    cors: {
      origin: SOCKET_CORS,
      credentials: true,
    },
  });

  // Subscribe to typing, message, and reaction events
  sub.psubscribe("typing:*", "message:*", "reaction:*");
  sub.on("pmessage", (_pattern, channel, message) => {
    console.log(`[Redis] Received pmessage on channel: ${channel}`);
    try {
      if (channel.startsWith("typing:")) {
        const payload: TypingPayload = JSON.parse(message);
        console.log(`[Socket] Broadcasting typing to room ${payload.conversationId} from user ${payload.userId}`);
        io.to(payload.conversationId).emit("typing", payload);
      } else if (channel.startsWith("message:")) {
        const payload: MessagePayload = JSON.parse(message);
        io.to(payload.conversationId).emit("newMessage", payload);
        console.log(`[Socket] Broadcasting newMessage to room ${payload.conversationId}`);
      } else if (channel.startsWith("reaction:")) {
        const payload: ReactionPayload = JSON.parse(message);
        io.to(payload.conversationId).emit("reactionUpdate", payload);
        console.log(`[Socket] Broadcasting reactionUpdate to room ${payload.conversationId}`);
      }
    } catch (err) {
      console.error("Failed to parse message", err);
    }
  });

  io.use(async (socket, next) => {
    const session = await getSessionFromSocket(socket);
    if (!session?.user?.id) {
      return next(new Error("unauthorized"));
    }
    (socket as AuthenticatedSocket).userId = session.user.id;
    (socket as AuthenticatedSocket).userName = session.user.name ?? undefined;
    next();
  });

  io.on("connection", async (socket) => {
    const authSocket = socket as AuthenticatedSocket;
    const userId: string = authSocket.userId;
    const name: string | undefined = authSocket.userName;

    console.log(`[Socket] New connection: user=${userId} name=${name} socketId=${socket.id}`);

    try {
      const convIds = await userConversationIds(userId);
      console.log(`[Socket] User ${userId} joining ${convIds.length} rooms:`, convIds);
      convIds.forEach((id) => socket.join(id));
    } catch (err) {
      console.error("Failed to join rooms", err);
    }

    socket.on("disconnect", (reason) => {
      console.log(`[Socket] Disconnected: user=${userId} socketId=${socket.id} reason=${reason}`);
    });

    socket.on("join", async ({ conversationId }: { conversationId?: string }) => {
      if (!conversationId) return;
      const isParticipant = await userIsParticipant(userId, conversationId);
      if (!isParticipant) return;
      socket.join(conversationId);
    });

    socket.on("typing", async ({ conversationId }: { conversationId?: string }) => {
      console.log(`[Socket] Received typing event from user ${userId} for conversation ${conversationId}`);
      if (!conversationId) {
        console.log("[Socket] No conversationId provided, ignoring typing event");
        return;
      }
      const isParticipant = await userIsParticipant(userId, conversationId);
      if (!isParticipant) {
        console.log(`[Socket] User ${userId} is not a participant in conversation ${conversationId}`);
        return;
      }
      const payload: TypingPayload = {
        conversationId,
        userId,
        name,
        expiresAt: Date.now() + 3000,
      };
      console.log(`[Redis] Publishing typing event to typing:${conversationId}`);
      pub.publish(`typing:${conversationId}`, JSON.stringify(payload));
    });
  });

  httpServer.listen(SOCKET_PORT, () => {
    console.log(`Socket server listening on ${SOCKET_PORT}`);
  });
}

main().catch((err) => {
  console.error("Socket server failed to start:", err);
  process.exit(1);
});
