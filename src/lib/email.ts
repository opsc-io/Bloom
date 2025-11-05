/**
 * Simple email sending abstraction using SMTP. Reads SMTP config from env.
 * Uses a dynamic require to avoid compile-time dependency on nodemailer types.
 */
const nodemailer = require('nodemailer');

const SMTP_HOST = process.env.SMTP_HOST || '';
const SMTP_PORT = process.env.SMTP_PORT ? parseInt(process.env.SMTP_PORT, 10) : undefined;
const SMTP_USER = process.env.SMTP_USER || '';
const SMTP_PASS = process.env.SMTP_PASS || '';
const SMTP_FROM = process.env.SMTP_FROM || 'no-reply@example.com';

let transporter: any = null;

function getTransporter() {
  if (transporter) return transporter;
  if (!SMTP_HOST) {
    // fallback: create a stub transporter that logs to console
    transporter = {
      sendMail: async (opts: any) => {
        console.log('Email send (stub):', opts);
        return { accepted: [opts.to] };
      },
    };
    return transporter;
  }

  transporter = nodemailer.createTransport({
    host: SMTP_HOST,
    port: SMTP_PORT || 587,
    secure: SMTP_PORT === 465,
    auth: SMTP_USER ? { user: SMTP_USER, pass: SMTP_PASS } : undefined,
  });
  return transporter;
}

export async function sendResetEmail(to: string, token: string) {
  const t = getTransporter();
  const appUrl = process.env.APP_URL || '';
  const resetLink = `${appUrl}/auth/password-reset?token=${token}`;
  const mail = {
    from: SMTP_FROM,
    to,
    subject: 'Password reset',
    text: `You requested a password reset. Click the link to reset your password: ${resetLink}`,
    html: `<p>You requested a password reset.</p><p><a href="${resetLink}">Reset your password</a></p>`,
  };
  return t.sendMail(mail);
}

export default { sendResetEmail };
