/**
 * Small user service helpers to find users and update credentials.
 */
import { AppDataSource } from '../db/data-source';
import { User } from '../db/entities/User';

export async function findUserByEmail(email: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(User);
  return repo.findOneBy({ email });
}

export async function findUserById(userId: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(User);
  return repo.findOneBy({ user_id: userId });
}

export default { findUserByEmail, findUserById };

export async function setPassword(userId: string, passwordHash: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(User);
  await repo.update({ user_id: userId }, { password_hash: passwordHash });
  return true;
}
