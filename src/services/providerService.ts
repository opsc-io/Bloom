/**
 * Provider service: manages OAuth provider link rows (user_providers table).
 */
import { AppDataSource } from '../db/data-source';
import { UserProvider } from '../db/entities/UserProvider';

export async function linkProvider(userId: string, provider: string, providerUserId: string, providerData: any) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(UserProvider);

  // Try to find an existing link
  let existing = await repo.findOneBy({ provider, provider_user_id: providerUserId });
  if (existing) {
    // If it exists but is attached to a different user, re-attach or throw depending on policy.
    if (existing.user_id !== userId) {
      existing.user_id = userId;
      existing.provider_data = providerData;
      await repo.save(existing);
    }
    return existing;
  }

  const link = new UserProvider();
  link.user_id = userId;
  link.provider = provider;
  link.provider_user_id = providerUserId;
  link.provider_data = providerData;
  return repo.save(link);
}

export default { linkProvider };

export async function listProviders(userId: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(UserProvider);
  return repo.find({ where: { user_id: userId }, order: { created_at: 'DESC' } });
}

export async function unlinkProvider(id: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const repo = AppDataSource.getRepository(UserProvider);
  const rec = await repo.findOneBy({ id });
  if (!rec) throw new Error('Provider link not found');
  await repo.remove(rec);
  return true;
}
