/**
 * NextAuth route for OAuth sign-in (Google + Facebook).
 *
 * This file uses NextAuth in the App Router (route handler). It implements a
 * signIn callback that finds or creates a user in our application's DB via
 * the existing authService. New OAuth signups default to the `client` role —
 * you can later implement an admin approval flow or provider-specific logic.
 */
import NextAuth from 'next-auth';
import authOptions from '../../../../src/auth/nextAuthOptions';

const handler = NextAuth(authOptions as any);

export { handler as GET, handler as POST };
