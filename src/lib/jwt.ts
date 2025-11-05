/**
 * Small JWT helper using jsonwebtoken. Exposes sign and verify helpers.
 */
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-change-me';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

export function signJwt(payload: object | string) {
	if (!JWT_SECRET) throw new Error('JWT secret is not configured');
	return jwt.sign(payload as any, JWT_SECRET as any, { expiresIn: JWT_EXPIRES_IN } as any);
}

export function verifyJwt<T = any>(token: string): T | null {
	try {
		return jwt.verify(token, JWT_SECRET as any) as T;
	} catch (err) {
		return null;
	}
}

export default { signJwt, verifyJwt };
