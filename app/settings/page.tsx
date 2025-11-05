import SetPasswordForm from './SetPasswordForm';
import { getServerSession } from 'next-auth';
import authOptions from '../../src/auth/nextAuthOptions';
import { redirect } from 'next/navigation';

export default async function SettingsPage() {
  const session = await getServerSession(authOptions as any);
  if (!session) redirect('/auth/login');

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <div className="mt-4">
        <SetPasswordForm />
      </div>
    </div>
  );
}
