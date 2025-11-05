import { NextResponse, NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { unlinkProvider } from '../../../../src/services/providerService';

export async function POST(req: NextRequest) {
  try {
    const token: any = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
    if (!token || !token.sub) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    const body = await req.json();
    const { id } = body as { id?: string };
    if (!id) return NextResponse.json({ error: 'Missing id' }, { status: 400 });
    await unlinkProvider(id);
    return NextResponse.json({ ok: true });
  } catch (err: any) {
    return NextResponse.json({ error: err.message || 'Failed' }, { status: 500 });
  }
}
