import Redis from 'ioredis';
import crypto from 'crypto';
import { findUserByEmail, findUserById } from './userService';

// TTL for reset tokens (seconds)
const TOKEN_TTL = parseInt(process.env.PASSWORD_RESET_TTL ?? '3600', 10);

type Storage = {
  set(key: string, value: string, ttlSeconds: number): Promise<void>;
  get(key: string): Promise<string | null>;
  del(key: string): Promise<void>;
};

class RedisStorage implements Storage {
  client: Redis;
  constructor(url?: string) {
    if (url) this.client = new Redis(url);
    else this.client = new Redis();
  }
  async set(key: string, value: string, ttlSeconds: number) {
    await this.client.set(key, value, 'EX', ttlSeconds);
  }
  async get(key: string) {
    return await this.client.get(key);
  }
  async del(key: string) {
    await this.client.del(key);
  }
}

class MemoryStorage implements Storage {
  store = new Map<string, { val: string; expiresAt: number }>();
  async set(key: string, value: string, ttlSeconds: number) {
    this.store.set(key, { val: value, expiresAt: Date.now() + ttlSeconds * 1000 });
  }
  async get(key: string) {
    const e = this.store.get(key);
    if (!e) return null;
    if (e.expiresAt < Date.now()) {
      this.store.delete(key);
      return null;
    }
    return e.val;
  }
  async del(key: string) {
    this.store.delete(key);
  }
}

const storage: Storage = ((): Storage => {
  if (process.env.REDIS_URL) return new RedisStorage(process.env.REDIS_URL);
  return new MemoryStorage();
})();

export async function generateResetTokenForEmail(email: string) {
  const user = await findUserByEmail(email);
  if (!user) return null;
  const token = crypto.randomBytes(24).toString('hex');
  const key = `pwdreset:${token}`;
  await storage.set(key, user.user_id, TOKEN_TTL);
  return { token, userId: user.user_id };
}

export async function verifyAndConsumeToken(token: string) {
  const key = `pwdreset:${token}`;
  const userId = await storage.get(key);
  if (!userId) return null;
  // consume
  await storage.del(key);
  const user = await findUserById(userId);
  return user ?? null;
}

export default { generateResetTokenForEmail, verifyAndConsumeToken };
