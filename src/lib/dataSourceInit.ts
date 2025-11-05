/**
 * Helper to ensure TypeORM DataSource is initialized before use in API handlers.
 */
import { AppDataSource } from '../db/data-source';

export async function ensureDataSource() {
  if (!AppDataSource.isInitialized) {
    await AppDataSource.initialize();
  }
}

export default ensureDataSource;
