import type { NextApiRequest, NextApiResponse } from 'next';
import { initRegistration } from '../../../../src/auth/webauthn';
import { findUserByEmail } from '../../../../src/services/userService';

// POST /api/auth/passkey/init-registration
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: 'Missing email' });

  const user = await findUserByEmail(email);
  if (!user) return res.status(404).json({ error: 'User not found' });

  try {
    const options = await initRegistration(user);
    return res.status(200).json(options);
  } catch (err: any) {
    return res.status(500).json({ error: err.message || 'Failed to init registration' });
  }
}
