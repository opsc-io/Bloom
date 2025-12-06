import { betterAuth } from 'better-auth'
import { prismaAdapter } from 'better-auth/adapters/prisma'
import prisma from '@/lib/prisma'
import { twoFactor } from 'better-auth/plugins'
import { createTransport } from 'nodemailer'

// Async email sender that loads env vars at call time
async function sendEmail(options: {
  to: string
  subject: string
  html: string
}) {
  // Read env vars at runtime, not at module load time
  const smtpUser = process.env.SMTP_USER
  const smtpPass = process.env.SMTP_PASSWORD

  if (!smtpUser || !smtpPass) {
    throw new Error('SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.')
  }

  const transporter = createTransport({
    host: 'mail.smtp2go.com',
    port: 2525,
    secure: false,
    auth: {
      user: smtpUser,
      pass: smtpPass,
    },
  })

  return transporter.sendMail({
    from: '"Bloom Health" <noreply@bloomhealth.us>',
    ...options,
  })
}


// Handle dynamic Vercel URLs for preview deployments
const getBaseURL = () => {
  if (process.env.BETTER_AUTH_URL) {
    return process.env.BETTER_AUTH_URL
  }
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`
  }
  return 'http://localhost:3000'
}

export const auth = betterAuth({
  secret: process.env.BETTER_AUTH_SECRET,
  experimental: { joins: true },
  baseURL: getBaseURL(),
  database: prismaAdapter(prisma, {
    provider: 'postgresql',
  }),
  trustedOrigins: [
    'http://localhost:3000',
    'https://bloomhealth.us',
    'https://www.bloomhealth.us',
    'https://qa.bloomhealth.us',
    process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : '',
  ].filter(Boolean),
  emailAndPassword: {
    enabled: true,
    disableSignUp: false,
    requireEmailVerification: true,
    minPasswordLength: 8,
    maxPasswordLength: 128,
    autoSignIn: true,
    // Using Better Auth's default scrypt-based hashing (from better-auth/crypto)
    // This is compatible with Vercel Edge Runtime and matches seed-admin.ts
    sendResetPassword: async ({ user, url }) => {
      await sendEmail({
        to: user.email,
        subject: 'Reset your Bloom Health password',
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #333;">Reset Your Password</h2>
            <p>Hi ${user.name || 'there'},</p>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <p style="margin: 30px 0;">
              <a href="${url}" style="background-color: #7c3aed; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                Reset Password
              </a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p style="color: #666; word-break: break-all;">${url}</p>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;" />
            <p style="color: #999; font-size: 12px;">Bloom Health</p>
          </div>
        `,
      })
    },
  },
  user: {
    additionalFields: {
      firstname: {
        type: 'string',
        required: true,
        input: true,
      },
      lastname: {
        type: 'string',
        input: true,
      },
      role: {
        type: 'string',
        input: false,
        default: 'UNSET',
      },
      administrator: {
        type: 'boolean',
        input: false,
        default: false,
      },
    },
  },
  socialProviders: {
    google: {
      accessType: "offline",
      prompt: "select_account consent",
      clientId: process.env.GOOGLE_CLIENT_ID as string,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
      mapProfileToUser: (profile) => ({
        firstname: profile.given_name,
        lastname: profile.family_name ? profile.family_name : " ",
        avatarUrl: profile.picture,
        email: profile.email,
      }),
    },
    zoom: {
      clientId: process.env.ZOOM_CLIENT_ID as string,
      clientSecret: process.env.ZOOM_CLIENT_SECRET as string,
    },
  },
  emailVerification: {
    sendOnSignUp: true,
    autoSignInAfterVerification: true,
    sendVerificationEmail: async ({ user, url }) => {
      await sendEmail({
        to: user.email,
        subject: 'Verify your Bloom Health email',
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #333;">Verify Your Email</h2>
            <p>Hi ${user.name || 'there'},</p>
            <p>Thanks for signing up for Bloom Health! Please verify your email address by clicking the button below:</p>
            <p style="margin: 30px 0;">
              <a href="${url}" style="background-color: #7c3aed; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
                Verify Email
              </a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p style="color: #666; word-break: break-all;">${url}</p>
            <p>This link will expire in 24 hours.</p>
            <p>If you didn't create an account, you can safely ignore this email.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;" />
            <p style="color: #999; font-size: 12px;">Bloom Health</p>
          </div>
        `,
      })
    },
  },
  plugins: [
    twoFactor({
      schema: {
        user: {
          fields: {
            twoFactorEnabled: "twoFactorEnabled",
          },
        },
        twoFactor: {
          modelName: "twoFactor",
          fields: {
            secret: "secret",
            backupCodes: "backupCodes",
            userId: "userId",
          },
        },
      },
      otpOptions: {
        async sendOTP({ user, otp }) {
          await sendEmail({
            to: user.email,
            subject: 'Your Bloom Health verification code',
            html: `
              <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #333;">Verification Code</h2>
                <p>Hi ${user.name || 'there'},</p>
                <p>Your verification code is:</p>
                <p style="font-size: 32px; font-weight: bold; color: #7c3aed; letter-spacing: 4px; margin: 30px 0;">
                  ${otp}
                </p>
                <p>This code will expire in 5 minutes.</p>
                <p>If you didn't request this code, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;" />
                <p style="color: #999; font-size: 12px;">Bloom Health</p>
              </div>
            `,
          })
        },
      },
    }),
  ],
})
