import { vi, describe, it, expect, beforeEach } from 'vitest';

// Mock next-auth jwt getToken
vi.mock('next-auth/jwt', () => ({
  getToken: vi.fn(),
}));

// Mock provider service
vi.mock('../src/services/providerService', () => ({
  listProviders: vi.fn(),
  unlinkProvider: vi.fn(),
}));

import { getToken } from 'next-auth/jwt';
import { listProviders, unlinkProvider } from '../src/services/providerService';
import { GET as listGET } from '../app/api/providers/list/route';
import { POST as unlinkPOST } from '../app/api/providers/unlink/route';

beforeEach(() => {
  vi.clearAllMocks();
});

describe('Providers API routes', () => {
  it('GET /app/api/providers/list - returns 401 when no session', async () => {
    (getToken as any).mockResolvedValue(null);
    const res: any = await listGET({} as any);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body).toHaveProperty('error');
  });

  it('GET /app/api/providers/list - returns providers when session present', async () => {
    (getToken as any).mockResolvedValue({ sub: 'user-123' });
    (listProviders as any).mockResolvedValue([{ id: 'p1', provider: 'google' }]);

    const res: any = await listGET({} as any);
    expect(listProviders).toHaveBeenCalledWith('user-123');
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
    expect(body[0]).toHaveProperty('id', 'p1');
  });

  it('POST /app/api/providers/unlink - returns 401 when no session', async () => {
    (getToken as any).mockResolvedValue(null);
    const req = { json: async () => ({ id: 'p1' }) } as any;
    const res: any = await unlinkPOST(req);
    expect(res.status).toBe(401);
  });

  it('POST /app/api/providers/unlink - calls unlinkProvider when session present', async () => {
    (getToken as any).mockResolvedValue({ sub: 'user-123' });
    (unlinkProvider as any).mockResolvedValue(true);
    const req = { json: async () => ({ id: 'p1' }) } as any;
    const res: any = await unlinkPOST(req);
    expect(unlinkProvider).toHaveBeenCalledWith('p1');
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty('ok', true);
  });
});
