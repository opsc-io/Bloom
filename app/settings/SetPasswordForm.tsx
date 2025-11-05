"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function SetPasswordForm() {
  const router = useRouter();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ok, setOk] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (newPassword !== confirmPassword) return setError('Passwords do not match');
    setLoading(true);
    try {
      const res = await fetch('/api/settings/password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ currentPassword, newPassword }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body?.error || 'Failed to set password');
      } else {
        setOk(true);
        // small delay then refresh
        setTimeout(() => router.refresh(), 600);
      }
    } catch (err: any) {
      setError(err?.message || 'Network error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={submit} className="max-w-md space-y-4">
      <h2 className="text-xl">Set or update your password</h2>
      {error && <div className="alert alert-error">{error}</div>}
      {ok && <div className="alert alert-success">Password updated</div>}

      <input type="password" placeholder="Current password (if any)" value={currentPassword} onChange={e=>setCurrentPassword(e.target.value)} className="input input-bordered w-full" />
      <input type="password" placeholder="New password" value={newPassword} onChange={e=>setNewPassword(e.target.value)} className="input input-bordered w-full" />
      <input type="password" placeholder="Confirm new password" value={confirmPassword} onChange={e=>setConfirmPassword(e.target.value)} className="input input-bordered w-full" />

      <div className="flex gap-2">
        <button className="btn btn-primary" type="submit" disabled={loading}>{loading ? 'Saving...' : 'Save password'}</button>
      </div>
    </form>
  );
}
