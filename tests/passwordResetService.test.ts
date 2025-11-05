import { vi, describe, it, expect, beforeEach } from 'vitest';

// Mock userService to avoid hitting TypeORM in this unit test.
vi.mock('../src/services/userService', () => ({
  findUserByEmail: vi.fn(),
  findUserById: vi.fn(),
}));

import { generateResetTokenForEmail, verifyAndConsumeToken } from '../src/services/passwordResetService';
import { findUserByEmail, findUserById } from '../src/services/userService';

beforeEach(() => { vi.clearAllMocks(); });

describe('passwordResetService', () => {
  it('generates and verifies tokens', async () => {
    const email = `test+${Date.now()}@example.com`;
    const fakeUser = { user_id: `u-${Date.now()}`, email };
    (findUserByEmail as any).mockResolvedValue(fakeUser);
    (findUserById as any).mockImplementation(async (id: string) => (id === fakeUser.user_id ? fakeUser : null));

    const res = await generateResetTokenForEmail(email);
    expect(res).not.toBeNull();
    const token = res!.token;

    const user = await verifyAndConsumeToken(token);
    expect(user).not.toBeNull();
    expect((user as any).email).toBe(email);

    // token should be consumed
    const again = await verifyAndConsumeToken(token);
    expect(again).toBeNull();
  });
});
