import { NextAuthOptions } from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';
import FacebookProvider from 'next-auth/providers/facebook';
import CredentialsProvider from 'next-auth/providers/credentials';
import { findUserByEmail } from '../services/userService';
import { signup, login } from '../services/authService';
import { linkProvider } from '../services/providerService';

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'text' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials) return null;
        try {
          const { email, password } = credentials as any;
          const res = await login(email, password);
          // login returns { token, user }
          return res.user as any;
        } catch (err) {
          return null;
        }
      },
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
    }),
    FacebookProvider({
      clientId: process.env.FACEBOOK_CLIENT_ID || '',
      clientSecret: process.env.FACEBOOK_CLIENT_SECRET || '',
    }),
  ],
  secret: process.env.NEXTAUTH_SECRET || process.env.JWT_SECRET || 'dev-nextauth-secret',
  session: { strategy: 'jwt' },
  callbacks: {
    async signIn({ user, account, profile }) {
      try {
        // When signing in via OAuth providers, ensure user exists and link provider
        if (account && account.provider) {
          let existing = await findUserByEmail(user.email as string);
          if (!existing) {
            existing = await signup({ email: user.email as string, accountType: 'client' }) as any;
          }
          try {
            await linkProvider((existing as any).user_id, account.provider, (account as any).providerAccountId, { profile, account });
          } catch (e) {
            console.warn('provider link failed:', e);
          }
        }
        return true;
      } catch (err) {
        console.error('NextAuth signIn callback error:', err);
        return false;
      }
    },
    async jwt({ token, user }) {
      // Attach user id to token for session
      if (user && (user as any).user_id) {
        token.sub = (user as any).user_id;
      }
      return token;
    },
    async session({ session, token }) {
      (session as any).user = { ...(session as any).user, user_id: token.sub };
      return session;
    },
  },
};

export default authOptions;
