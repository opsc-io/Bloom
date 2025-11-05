/**
 * WebAuthn storage adapter.
 * - Challenges are stored in Redis with a short TTL.
 * - Credentials are persisted on the User entity (jsonb column) via TypeORM.
 *
 * This file centralizes storage so `webauthn.ts` can focus on crypto/validation.
 */
import { redisSet, redisGet, redisDel } from '../lib/redis';
import { AppDataSource } from '../db/data-source';
import { User } from '../db/entities/User';

const CHALLENGE_PREFIX = 'webauthn:challenge:';
const CHALLENGE_TTL = 60 * 5; // 5 minutes

export async function storeChallenge(userId: string, challenge: string) {
  await redisSet(CHALLENGE_PREFIX + userId, challenge, CHALLENGE_TTL);
}

export async function getChallenge(userId: string) {
  return redisGet(CHALLENGE_PREFIX + userId);
}

export async function clearChallenge(userId: string) {
  await redisDel(CHALLENGE_PREFIX + userId);
}

export async function storeCredential(userId: string, credential: any) {
  // Persist passkey_credential into the users table's jsonb column
  const repo = AppDataSource.getRepository(User);
  const user = await repo.findOneBy({ user_id: userId });
  if (!user) throw new Error('User not found');
  user.passkey_credential = credential;
  await repo.save(user);
}

export async function updateCredentialCounter(userId: string, newCounter: number) {
  const repo = AppDataSource.getRepository(User);
  const user = await repo.findOneBy({ user_id: userId });
  if (!user) throw new Error('User not found');
  if (!user.passkey_credential) throw new Error('User has no stored credential');
  user.passkey_credential.counter = newCounter;
  await repo.save(user);
}

export async function getUserCredential(userId: string) {
  const repo = AppDataSource.getRepository(User);
  const user = await repo.findOneBy({ user_id: userId });
  return user?.passkey_credential;
}
