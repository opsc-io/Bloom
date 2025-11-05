"use client";
import React, { useEffect, useState } from 'react';
import { signIn } from 'next-auth/react';

export default function ProvidersPage() {
  const [providers, setProviders] = useState<any[]>([]);

  async function fetchProviders() {
    const res = await fetch('/api/providers/list');
    if (res.ok) setProviders(await res.json());
  }

  useEffect(() => { fetchProviders(); }, []);

  return (
    <div className="space-y-4">
      <h2 className="text-xl">Connected Providers</h2>
      <ul>
        {providers.map((p:any)=> (
          <li key={p.id} className="flex justify-between items-center py-2">
            <div>{p.provider} — {p.provider_user_id}</div>
            <div>
              <button className="btn btn-sm btn-error" onClick={async ()=>{
                await fetch('/api/providers/unlink', { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify({ id: p.id }) });
                fetchProviders();
              }}>Unlink</button>
            </div>
          </li>
        ))}
      </ul>

      <div className="divider" />

      <h3 className="text-lg">Link another provider</h3>
      <div className="flex gap-2">
        <button className="btn btn-outline" onClick={() => signIn('google', { callbackUrl: '/dashboard' })}>Link Google</button>
        <button className="btn btn-outline" onClick={() => signIn('facebook', { callbackUrl: '/dashboard' })}>Link Facebook</button>
      </div>
    </div>
  );
}
