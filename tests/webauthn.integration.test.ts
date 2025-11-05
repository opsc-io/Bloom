import { vi, describe, it, expect, beforeEach } from 'vitest';

// We'll run an integration-style test for high-level WebAuthn flows by
// importing the real `webauthn.ts` module but replacing the lower-level
// storage adapter with an in-memory stand-in and mocking the
// `@simplewebauthn/server` functions to produce deterministic results.

vi.mock('@simplewebauthn/server', () => ({
  generateRegistrationOptions: vi.fn(() => ({ challenge: 'reg-ch', rp: { name: 'Test' } })),
  verifyRegistrationResponse: vi.fn(async () => ({ verified: true, registrationInfo: { credentialPublicKey: 'pub', credentialID: 'cid', counter: 0 } })),
  generateAuthenticationOptions: vi.fn(() => ({ challenge: 'auth-ch' })),
  verifyAuthenticationResponse: vi.fn(async () => ({ verified: true, authenticationInfo: { newCounter: 1 } })),
}));

// Replace storage adapter with an in-memory implementation for this test.
const inMemoryStore: Record<string, any> = {};
vi.mock('../src/services/webauthnStorage', () => ({
  storeChallenge: vi.fn(async (userId: string, challenge: string) => { inMemoryStore[`ch:${userId}`] = challenge; }),
  getChallenge: vi.fn(async (userId: string) => inMemoryStore[`ch:${userId}`]),
  clearChallenge: vi.fn(async (userId: string) => { delete inMemoryStore[`ch:${userId}`]; }),
  storeCredential: vi.fn(async (userId: string, credential: any) => { inMemoryStore[`cred:${userId}`] = credential; }),
  updateCredentialCounter: vi.fn(async (userId: string, newCounter: number) => { if (inMemoryStore[`cred:${userId}`]) inMemoryStore[`cred:${userId}`].counter = newCounter; }),
  getUserCredential: vi.fn(async (userId: string) => inMemoryStore[`cred:${userId}`] || null),
}));

import { initRegistration, verifyRegistration, initAuthentication, verifyAuthentication } from '../src/auth/webauthn';

beforeEach(() => {
  vi.clearAllMocks();
  for (const k of Object.keys(inMemoryStore)) delete inMemoryStore[k];
});

describe('WebAuthn integration-style flow', () => {
  it('register -> verify -> authenticate -> verify works end-to-end', async () => {
    const fakeUser: any = { user_id: 'u-int', email: 'int@local', passkey_credential: null };

    // init registration
    const regOptions = await initRegistration(fakeUser);
    expect(regOptions).toHaveProperty('challenge', 'reg-ch');

    // verify registration (this should store credential via storage adapter)
    const regResp = { id: 'cid', rawId: 'cid', response: {}, type: 'public-key' } as any;
    const ok = await verifyRegistration(fakeUser, regResp as any);
    expect(ok).toBe(true);
    const stored = await (await import('../src/services/webauthnStorage')).getUserCredential(fakeUser.user_id);
    expect(stored).not.toBeNull();

    // init authentication
    const authOptions = await initAuthentication({ ...fakeUser, passkey_credential: stored });
    expect(authOptions).toHaveProperty('challenge', 'auth-ch');

    // verify authentication
    const authResp = { id: 'cid', rawId: 'cid', response: {}, type: 'public-key' } as any;
    const authOk = await verifyAuthentication({ ...fakeUser, passkey_credential: stored }, authResp as any);
    expect(authOk).toBe(true);
    const updated = await (await import('../src/services/webauthnStorage')).getUserCredential(fakeUser.user_id);
    expect(updated.counter).toBe(1);
  });
});
