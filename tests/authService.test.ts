import 'reflect-metadata';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// We'll mock AppDataSource to avoid a real DB connection and test signup happy path.
vi.mock('../src/db/data-source', () => ({
  AppDataSource: {
    isInitialized: true,
    getRepository: () => ({
      findOneBy: async (q: any) => null,
      save: async (entity: any) => ({ ...entity, user_id: 'mock-uuid' }),
    }),
  },
}));

// Mock TypeORM entities to avoid decorator evaluation in the test environment
vi.mock('../src/db/entities/Role', () => ({ Role: class Role { role_id = 0; name = ''; } }));
vi.mock('../src/db/entities/User', () => ({ User: class User { user_id = ''; email = ''; is_active = true; password_hash?: string } }));
vi.mock('../src/db/entities/UserRole', () => ({ UserRole: class UserRole { user_id = ''; role_id = 0; } }));

import { signup } from '../src/services/authService';

describe('authService.signup', () => {
  it('creates a provider account', async () => {
    const user = await signup({ email: 'p@example.com', password: 'secret', accountType: 'provider' });
    expect(user).toHaveProperty('email', 'p@example.com');
    expect(user).toHaveProperty('user_id');
  });
});
