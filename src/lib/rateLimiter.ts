/**
 * Lightweight rate limiter with pluggable backend.
 * Provides a simple token-bucket style limiter keyed by a string (IP or userId).
 * Default in-memory implementation is good for low-traffic or testing; Redis can be used in production.
 */
import Redis from 'ioredis';

type Backend = {
  incr(key: string, ttlSeconds: number): Promise<number>;
};

class MemoryBackend implements Backend {
  private map = new Map<string, { count: number; expiresAt: number }>();

  async incr(key: string, ttlSeconds: number) {
    const now = Date.now();
    const entry = this.map.get(key);
    if (!entry || entry.expiresAt < now) {
      this.map.set(key, { count: 1, expiresAt: now + ttlSeconds * 1000 });
      return 1;
    }
    entry.count += 1;
    return entry.count;
  }
}

class RedisBackend implements Backend {
  private client: Redis;
  constructor(client: Redis) {
    this.client = client;
  }
  async incr(key: string, ttlSeconds: number) {
    const val = await this.client.multi().incr(key).expire(key, ttlSeconds).exec();
    // val is array; take first result of incr
    const incrRes = val?.[0]?.[1] as number | null;
    return incrRes ?? 0;
  }
}

const backend: Backend = ((): Backend => {
  if (process.env.REDIS_URL) {
    try {
      const client = new Redis(process.env.REDIS_URL as string);
      return new RedisBackend(client);
    } catch (e) {
      // fallthrough to memory
    }
  }
  return new MemoryBackend();
})();

/**
 * Increment a key and return current count after increment. TTL is applied on first increment.
 * Returns true if allowed (count <= limit), false if over limit.
 */
export async function checkRate(key: string, limit = 5, windowSeconds = 60) {
  const count = await backend.incr(key, windowSeconds);
  return { allowed: count <= limit, count };
}

export default { checkRate };
