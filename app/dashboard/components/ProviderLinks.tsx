"use client";
import React, { useEffect, useState } from 'react';
import { signIn } from 'next-auth/react';

export default function ProviderLinks() {
  const [providers, setProviders] = useState<any[]>([]);

  async function fetchProviders() {
    const res = await fetch('/api/providers/list');
    if (res.ok) setProviders(await res.json());
  }

  useEffect(() => { fetchProviders(); }, []);

  return (
    <div className="card bg-base-100 shadow">
      <div className="card-body">
        <h3 className="card-title">Connected providers</h3>
        <ul className="mt-2 space-y-2">
          {providers.length === 0 && <li className="text-sm text-muted">No providers linked.</li>}
          {providers.map((p:any)=> (
            <li key={p.id} className="flex justify-between items-center">
              <div>{p.provider} — {p.provider_user_id}</div>
              <button className="btn btn-xs btn-error" onClick={async ()=>{
                await fetch('/api/providers/unlink', { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify({ id: p.id }) });
                fetchProviders();
              }}>Unlink</button>
            </li>
          ))}
        </ul>

        <div className="divider" />
        <div className="flex gap-2">
          <button className="btn btn-outline" onClick={() => signIn('google', { callbackUrl: '/dashboard' })}>Link Google</button>
          <button className="btn btn-outline" onClick={() => signIn('facebook', { callbackUrl: '/dashboard' })}>Link Facebook</button>
        </div>
      </div>
    </div>
  );
}
