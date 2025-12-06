import Redis from "ioredis";

const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";

const globalForRedis = global as unknown as {
  redis: Redis | undefined;
};

const redis = globalForRedis.redis ?? new Redis(REDIS_URL, {
  maxRetriesPerRequest: 3,
  retryStrategy(times) {
    if (times > 3) return null;
    return Math.min(times * 100, 3000);
  },
});

if (process.env.NODE_ENV !== "production") globalForRedis.redis = redis;

export default redis;

// Cache key patterns
export const CACHE_KEYS = {
  userConversations: (userId: string) => `conversations:${userId}`,
  conversationMessages: (conversationId: string) => `messages:${conversationId}`,
  conversationParticipants: (conversationId: string) => `participants:${conversationId}`,
} as const;

// Cache TTLs in seconds
export const CACHE_TTL = {
  conversations: 60, // 1 minute
  messages: 30, // 30 seconds
  participants: 300, // 5 minutes
} as const;

// Helper to get cached data or fetch from source
export async function getCachedOrFetch<T>(
  key: string,
  ttl: number,
  fetchFn: () => Promise<T>
): Promise<T> {
  try {
    const cached = await redis.get(key);
    if (cached) {
      console.log(`[Redis] CACHE HIT: ${key}`);
      return JSON.parse(cached) as T;
    }
    console.log(`[Redis] CACHE MISS: ${key}`);
  } catch (err) {
    console.error(`[Redis] GET error for ${key}:`, err);
  }

  const data = await fetchFn();

  try {
    await redis.setex(key, ttl, JSON.stringify(data));
    console.log(`[Redis] SET: ${key} (TTL: ${ttl}s)`);
  } catch (err) {
    console.error(`[Redis] SET error for ${key}:`, err);
  }

  return data;
}

// Invalidate cache for a user's conversations
export async function invalidateUserConversations(userId: string): Promise<void> {
  try {
    await redis.del(CACHE_KEYS.userConversations(userId));
  } catch {
    // Ignore Redis errors
  }
}

// Invalidate cache for a conversation's messages
export async function invalidateConversationMessages(conversationId: string): Promise<void> {
  try {
    await redis.del(CACHE_KEYS.conversationMessages(conversationId));
  } catch {
    // Ignore Redis errors
  }
}

// Invalidate all caches related to a conversation (for both participants)
export async function invalidateConversation(
  conversationId: string,
  participantIds: string[]
): Promise<void> {
  try {
    const keys = [
      CACHE_KEYS.conversationMessages(conversationId),
      ...participantIds.map((id) => CACHE_KEYS.userConversations(id)),
    ];
    if (keys.length > 0) {
      await redis.del(...keys);
    }
  } catch {
    // Ignore Redis errors
  }
}
