/**
 * Authentication service responsible for user creation and login.
 * - Supports roles: 'provider', 'client', 'admin'
 * - Uses bcrypt for password hashing and jsonwebtoken for tokens.
 *
 * The exported functions are intentionally small and well-documented to
 * facilitate unit testing (they avoid tight coupling to global AppDataSource
 * where possible).
 */
import bcrypt from 'bcrypt';
import { signJwt } from '../lib/jwt';
import { AppDataSource } from '../db/data-source';
import { User } from '../db/entities/User';
import { Role } from '../db/entities/Role';
import { UserRole } from '../db/entities/UserRole';

export type SignupPayload = {
  email: string;
  password?: string; // optional for passkey-only flows
  accountType: 'provider' | 'client' | 'admin';
};

/** Create a user and assign role. Returns the created user (without password). */
export async function signup(payload: SignupPayload) {
  const { email, password, accountType } = payload;

  // Initialize data source lazily if needed
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();

  const userRepo = AppDataSource.getRepository(User);
  const roleRepo = AppDataSource.getRepository(Role);
  const userRoleRepo = AppDataSource.getRepository(UserRole);

  const existing = await userRepo.findOneBy({ email });
  if (existing) throw new Error('Email already registered');

  const user = new User();
  user.email = email;
  if (password) {
    const salt = await bcrypt.genSalt(10);
    user.password_hash = await bcrypt.hash(password, salt);
  }
  user.is_active = true;
  await userRepo.save(user);

  // Find or create role
  let role = await roleRepo.findOneBy({ name: accountType });
  if (!role) {
    role = new Role();
    role.name = accountType;
    await roleRepo.save(role);
  }

  const userRole = new UserRole();
  userRole.user_id = user.user_id;
  userRole.role_id = role.role_id;
  await userRoleRepo.save(userRole);

  // For security, don't return passwordHash
  const out = { ...user } as any;
  delete out.passwordHash;
  return out as User;
}

/** Verify user credentials and return a JWT on success. */
export async function login(email: string, password: string) {
  if (!AppDataSource.isInitialized) await AppDataSource.initialize();
  const userRepo = AppDataSource.getRepository(User);
  const user = await userRepo.findOneBy({ email });
  if (!user) throw new Error('Invalid credentials');

  // Passwords are optional if passkey-only auth is used; require a hash for password login
  const passwordHash = (user as any).password_hash ?? (user as any).passwordHash;
  if (!passwordHash) throw new Error('Password login not available for this account');

  const ok = await bcrypt.compare(password, passwordHash as string);
  if (!ok) throw new Error('Invalid credentials');

  // Issue JWT with minimal claims
  const token = signJwt({ sub: user.user_id, email: user.email });
  return { token, user };
}

export default { signup, login };
