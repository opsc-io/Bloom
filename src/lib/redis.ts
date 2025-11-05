/**
 * Redis client singleton used across the server.
 * Uses ioredis which is already present in package.json.
 */
import Redis from 'ioredis';

const REDIS_HOST = process.env.REDIS_HOST || '127.0.0.1';
const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379', 10);

// Create a single Redis instance to reuse across modules
export const redis = new Redis({ host: REDIS_HOST, port: REDIS_PORT });

// Helper wrappers (small convenience functions)
export async function redisSet(key: string, value: string, ttlSeconds?: number) {
  if (ttlSeconds) {
    await redis.set(key, value, 'EX', ttlSeconds);
  } else {
    await redis.set(key, value);
  }
}

export async function redisGet(key: string) {
  return redis.get(key);
}

export async function redisDel(key: string) {
  return redis.del(key);
}

export default redis;
