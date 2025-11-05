import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('../src/services/userService', () => ({
  findUserByEmail: vi.fn(),
}));

vi.mock('../src/services/authService', () => ({
  signup: vi.fn(),
  login: vi.fn(),
}));

vi.mock('../src/services/providerService', () => ({
  linkProvider: vi.fn(),
}));

import authOptions from '../src/auth/nextAuthOptions';
import { findUserByEmail } from '../src/services/userService';
import { signup } from '../src/services/authService';
import { linkProvider } from '../src/services/providerService';

beforeEach(() => {
  vi.clearAllMocks();
});

describe('NextAuth signIn callback', () => {
  it('creates and links provider when user not found', async () => {
    (findUserByEmail as any).mockResolvedValue(null);
    (signup as any).mockResolvedValue({ user_id: 'u1', user: { user_id: 'u1' } });
    (linkProvider as any).mockResolvedValue(true);

    const user = { email: 'x@x.com' } as any;
    const account = { provider: 'google', providerAccountId: 'p1' } as any;
    const result = await (authOptions.callbacks as any).signIn({ user, account, profile: {} });
    expect(result).toBe(true);
    expect(signup).toHaveBeenCalled();
    expect(linkProvider).toHaveBeenCalled();
  });

  it('links provider when user exists', async () => {
    (findUserByEmail as any).mockResolvedValue({ user_id: 'u2' });
    (linkProvider as any).mockResolvedValue(true);

    const user = { email: 'y@y.com' } as any;
    const account = { provider: 'facebook', providerAccountId: 'p2' } as any;
    const result = await (authOptions.callbacks as any).signIn({ user, account, profile: {} });
    expect(result).toBe(true);
    expect(signup).not.toHaveBeenCalled();
    expect(linkProvider).toHaveBeenCalledWith('u2', 'facebook', (account as any).providerAccountId, expect.any(Object));
  });
});
