import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('../src/auth/webauthn', () => ({
  initRegistration: vi.fn(),
  verifyRegistration: vi.fn(),
  initAuthentication: vi.fn(),
  verifyAuthentication: vi.fn(),
}));

vi.mock('../src/services/userService', () => ({
  findUserByEmail: vi.fn(),
}));

import initRegHandler from '../pages/api/auth/passkey/init-registration';
import verifyRegHandler from '../pages/api/auth/passkey/verify-registration';
import initAuthHandler from '../pages/api/auth/passkey/init-authentication';
import verifyAuthHandler from '../pages/api/auth/passkey/verify-authentication';

import { initRegistration, verifyRegistration, initAuthentication, verifyAuthentication } from '../src/auth/webauthn';
import { findUserByEmail } from '../src/services/userService';

beforeEach(() => {
  vi.clearAllMocks();
});

function makeRes() {
  const body: any = {};
  const res: any = {
    status(code: number) {
      body.status = code;
      return res;
    },
    json(payload: any) {
      body.payload = payload;
      return res;
    },
  };
  return { res, body };
}

describe('WebAuthn API handlers', () => {
  it('init-registration returns 405 for non-POST', async () => {
  const { res, body } = makeRes();
  await initRegHandler({ method: 'GET' } as any, res);
  expect(body.status).toBe(405);
  });

  it('init-registration returns 400 when missing email', async () => {
  const { res, body } = makeRes();
  await initRegHandler({ method: 'POST', body: {} } as any, res);
  expect(body.status).toBe(400);
  });

  it('init-registration returns 404 when user not found', async () => {
    (findUserByEmail as any).mockResolvedValue(null);
  const { res, body } = makeRes();
  await initRegHandler({ method: 'POST', body: { email: 'x@x.com' } } as any, res);
  expect(body.status).toBe(404);
  });

  it('init-registration returns options for valid user', async () => {
    (findUserByEmail as any).mockResolvedValue({ user_id: 'u1' });
    (initRegistration as any).mockResolvedValue({ challenge: 'abc' });
  const { res, body } = makeRes();
  await initRegHandler({ method: 'POST', body: { email: 'x@x.com' } } as any, res);
  expect(body.status).toBe(200);
  expect(body.payload).toHaveProperty('challenge', 'abc');
  });

  it('verify-registration returns 400 for missing fields', async () => {
  const { res, body } = makeRes();
  await verifyRegHandler({ method: 'POST', body: {} } as any, res);
  expect(body.status).toBe(400);
  });

  it('verify-registration returns 404 when user missing', async () => {
    (findUserByEmail as any).mockResolvedValue(null);
  const { res, body } = makeRes();
  await verifyRegHandler({ method: 'POST', body: { email: 'x', response: {} } } as any, res);
  expect(body.status).toBe(404);
  });

  it('verify-registration returns verified true on success', async () => {
    (findUserByEmail as any).mockResolvedValue({ user_id: 'u1' });
    (verifyRegistration as any).mockResolvedValue(true);
  const { res, body } = makeRes();
  await verifyRegHandler({ method: 'POST', body: { email: 'x', response: {} } } as any, res);
  expect(body.status).toBe(200);
  expect(body.payload).toHaveProperty('verified', true);
  });

  it('init-authentication returns 405 for non-POST', async () => {
  const { res, body } = makeRes();
  await initAuthHandler({ method: 'GET' } as any, res);
  expect(body.status).toBe(405);
  });

  it('init-authentication returns 404 when user not found', async () => {
    (findUserByEmail as any).mockResolvedValue(null);
  const { res, body } = makeRes();
  await initAuthHandler({ method: 'POST', body: { email: 'x' } } as any, res);
  expect(body.status).toBe(404);
  });

  it('init-authentication returns options for valid user', async () => {
    (findUserByEmail as any).mockResolvedValue({ user_id: 'u1', passkey_credential: { id: 'c1' } });
    (initAuthentication as any).mockResolvedValue({ challenge: 'auth-ch' });
  const { res, body } = makeRes();
  await initAuthHandler({ method: 'POST', body: { email: 'x' } } as any, res);
  expect(body.status).toBe(200);
  expect(body.payload).toHaveProperty('challenge', 'auth-ch');
  });

  it('verify-authentication returns verified true on success', async () => {
    (findUserByEmail as any).mockResolvedValue({ user_id: 'u1', passkey_credential: { id: 'c1' } });
    (verifyAuthentication as any).mockResolvedValue(true);
  const { res, body } = makeRes();
  await verifyAuthHandler({ method: 'POST', body: { email: 'x', response: {} } } as any, res);
  expect(body.status).toBe(200);
  expect(body.payload).toHaveProperty('verified', true);
  });
});
