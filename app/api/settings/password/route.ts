import { NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import bcrypt from 'bcrypt';
import { setPassword as setUserPassword, findUserById } from '../../../../src/services/userService';

const SALT_ROUNDS = 10;

export async function POST(req: Request) {
  try {
  // `getToken` typing expects Next.js request types; cast to any for runtime.
  const token = await getToken({ req: req as any, secret: process.env.NEXTAUTH_SECRET || process.env.JWT_SECRET });
    if (!token || !token.sub) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const body = await req.json();
    const { currentPassword, newPassword } = body || {};
    if (!newPassword || typeof newPassword !== 'string' || newPassword.length < 6) {
      return NextResponse.json({ error: 'Invalid new password' }, { status: 400 });
    }

    const userId = token.sub as string;
    const user = await findUserById(userId);
    if (!user) return NextResponse.json({ error: 'User not found' }, { status: 404 });

    // If user already has a password, verify currentPassword
    if (user.password_hash) {
      if (!currentPassword) return NextResponse.json({ error: 'Current password required' }, { status: 400 });
      const match = await bcrypt.compare(currentPassword, user.password_hash);
      if (!match) return NextResponse.json({ error: 'Current password incorrect' }, { status: 403 });
    }

    const hashed = await bcrypt.hash(newPassword, SALT_ROUNDS);
    await setUserPassword(userId, hashed);
    return NextResponse.json({ ok: true });
  } catch (err: any) {
    console.error('password set error', err);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
