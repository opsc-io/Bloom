import type { NextApiRequest, NextApiResponse } from 'next';
import { verifyAuthentication } from '../../../../../src/auth/webauthn';
import { findUserByEmail } from '../../../../../src/services/userService';

// POST /api/auth/passkey/verify-authentication
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { email, response } = req.body;
  if (!email || !response) return res.status(400).json({ error: 'Missing email or response' });

  const user = await findUserByEmail(email);
  if (!user) return res.status(404).json({ error: 'User not found' });

  try {
    const ok = await verifyAuthentication(user, response);
    return res.status(200).json({ verified: !!ok });
  } catch (err: any) {
    return res.status(400).json({ error: err.message || 'Verification failed' });
  }
}
