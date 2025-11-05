import { getServerSession } from 'next-auth';
import authOptions from '../../src/auth/nextAuthOptions';
import { redirect } from 'next/navigation';
import dynamic from 'next/dynamic';

const DashboardCards = dynamic(() => import('./components/DashboardCards'));
const ProviderLinks = dynamic(() => import('./components/ProviderLinks'));

export default async function DashboardPage() {
  const session = await getServerSession(authOptions as any);
  if (!session) redirect('/auth/login');

  const user = (session as any).user || {};

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <p className="mt-1">Welcome, {user.email ?? user.name ?? 'user'}!</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <DashboardCards />
        <div className="md:col-span-2">
          <ProviderLinks />
        </div>
      </div>
    </div>
  );
}
