"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { signIn } from 'next-auth/react';

export default function LoginForm() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    // Use NextAuth credentials provider
    const result: any = await signIn('credentials', { redirect: false, email, password });
    if (result?.error) {
      setError(result.error || 'Login failed');
      return;
    }
    router.push('/');
  }

  return (
    <div>
      <form onSubmit={submit} className="space-y-4">
        <h2 className="text-2xl font-semibold">Log in</h2>
        {error && <div className="alert alert-error">{error}</div>}
        <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email" className="input input-bordered w-full" />
        <input value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password" type="password" className="input input-bordered w-full" />
        <button className="btn btn-primary w-full" type="submit">Sign in</button>
      </form>

      <div className="divider">OR</div>

      <div className="flex gap-2">
        <button className="btn btn-outline w-full" onClick={() => signIn('google', { callbackUrl: '/dashboard' })}>Sign in with Google</button>
        <button className="btn btn-outline w-full" onClick={() => signIn('facebook', { callbackUrl: '/dashboard' })}>Sign in with Facebook</button>
      </div>
    </div>
  );
}
