import { NextResponse } from 'next/server';
import { generateResetTokenForEmail } from '../../../../../src/services/passwordResetService';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const email = body?.email;
    if (!email) return NextResponse.json({ error: 'Missing email' }, { status: 400 });

    // Rate-limit password-reset requests per email to avoid spam
    const { checkRate } = await import('../../../../../src/lib/rateLimiter');
    const reqLimit = parseInt(process.env.PASSWORD_RESET_REQUEST_LIMIT ?? '3', 10);
    const reqWindow = parseInt(process.env.PASSWORD_RESET_REQUEST_WINDOW ?? '3600', 10);
    const rlr = await checkRate(`pwdreq:email:${email}`, reqLimit, reqWindow);
    if (!rlr.allowed) return NextResponse.json({ ok: true });

    const res = await generateResetTokenForEmail(email);
    if (!res) {
      // Do not reveal whether an account exists; return 200 to avoid account enumeration.
      return NextResponse.json({ ok: true });
    }

    // Send reset email using configured SMTP (falls back to console logging when not configured)
    try {
      const { sendResetEmail } = await import('../../../../../src/lib/email');
      await sendResetEmail(email, res.token);
    } catch (e) {
      // fallback: log the reset link
      const resetLink = `${process.env.APP_URL ?? ''}/auth/password-reset?token=${res.token}`;
      console.log(`Password reset for ${email}: ${resetLink}`);
    }

    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error('password reset request error', err);
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
