import { NextResponse } from 'next/server';
import bcrypt from 'bcrypt';
import { verifyAndConsumeToken } from '../../../../../src/services/passwordResetService';
import { setPassword } from '../../../../../src/services/userService';

const SALT_ROUNDS = 10;

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { token, newPassword } = body || {};
    if (!token || !newPassword) return NextResponse.json({ error: 'Missing fields' }, { status: 400 });
    if (typeof newPassword !== 'string' || newPassword.length < 6) return NextResponse.json({ error: 'Invalid password' }, { status: 400 });

    const user = await verifyAndConsumeToken(token);
    if (!user) return NextResponse.json({ error: 'Invalid or expired token' }, { status: 400 });

    const hash = await bcrypt.hash(newPassword, SALT_ROUNDS);
    await setPassword(user.user_id, hash);
    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error('password reset confirm error', err);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
