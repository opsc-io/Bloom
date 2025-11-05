/**
 * Simple migration runner that initializes the AppDataSource and runs pending migrations.
 * Use: npm run migrate
 */
import 'reflect-metadata';
import { AppDataSource } from '../src/db/data-source';

async function run() {
  try {
    if (!AppDataSource.isInitialized) await AppDataSource.initialize();
    console.log('Running migrations...');
    const res = await AppDataSource.runMigrations();
    console.log('Migrations complete:', res.map(r => r.name));
    await AppDataSource.destroy();
  } catch (err) {
    console.error('Migration failed:', err);
    process.exit(1);
  }
}

run();
