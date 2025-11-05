"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function SignupForm() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [accountType, setAccountType] = useState<'provider' | 'client'>('provider');
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const res = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, accountType }),
    });
    const data = await res.json();
    if (!res.ok) {
      setError(data.error || 'Signup failed');
      return;
    }
    router.push('/auth/login');
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <h2 className="text-2xl font-semibold">Sign up</h2>
      {error && <div className="alert alert-error">{error}</div>}
      <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email" className="input input-bordered w-full" />
      <input value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password" type="password" className="input input-bordered w-full" />
      <select value={accountType} onChange={e=>setAccountType(e.target.value as any)} className="select select-bordered w-full">
        <option value="provider">Provider</option>
        <option value="client">Client</option>
      </select>
      <button className="btn btn-primary w-full" type="submit">Create account</button>
    </form>
  );
}
