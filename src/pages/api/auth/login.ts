import type { NextApiRequest, NextApiResponse } from 'next';
import { login } from '../../../../src/services/authService';

// POST /api/auth/login
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Missing email or password' });

    const { token, user } = await login(email, password);
    return res.status(200).json({ token, user });
  } catch (err: any) {
    return res.status(401).json({ error: err.message || 'Login failed' });
  }
}
