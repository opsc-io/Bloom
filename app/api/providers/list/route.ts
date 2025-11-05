import { NextResponse, NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { listProviders } from '../../../../src/services/providerService';

export async function GET(req: NextRequest) {
  try {
    const token: any = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token || !token.sub) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    const providers = await listProviders(token.sub);
    return NextResponse.json(providers);
  } catch (err: any) {
    return NextResponse.json({ error: err.message || 'Failed' }, { status: 500 });
  }
}
