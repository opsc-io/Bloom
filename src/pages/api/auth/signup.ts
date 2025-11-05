import type { NextApiRequest, NextApiResponse } from 'next';
import { signup } from '../../../../src/services/authService';

// POST /api/auth/signup
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const { email, password, accountType } = req.body;
    if (!email || !accountType) return res.status(400).json({ error: 'Missing email or accountType' });

    // Only providers and clients can sign up directly. Admins require separate onboarding.
    if (accountType === 'admin') {
      return res.status(403).json({ error: 'Admin onboarding must be performed separately' });
    }

    const user = await signup({ email, password, accountType });
    return res.status(201).json({ user });
  } catch (err: any) {
    return res.status(400).json({ error: err.message || 'Signup failed' });
  }
}
