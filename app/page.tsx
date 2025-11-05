import Link from 'next/link';
export default function Page() {
  return (
    <div className="prose">
      <h1>Therapy Practice Platform</h1>
      <p>Welcome — manage your practice, appointments, and credentialing.</p>
      <div className="space-x-2">
        <Link href="/auth/signup" className="btn btn-primary">Sign up</Link>
        <Link href="/auth/login" className="btn btn-secondary">Log in</Link>
      </div>
    </div>
  );
}
