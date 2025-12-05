import { PrismaClient, UserRole } from '../src/generated/prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import { hashPassword } from 'better-auth/crypto';
import 'dotenv/config';

const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL!,
});

const prisma = new PrismaClient({ adapter });

async function seedAdmin() {
  const email = 'admin@bloomhealth.us';
  const password = 'Admin1234!';
  const hashedPassword = await hashPassword(password);
  const id = crypto.randomUUID();

  const admin = await prisma.user.upsert({
    where: { email },
    update: {
      role: UserRole.ADMINISTRATOR,
      administrator: true,
      emailVerified: true,
    },
    create: {
      id,
      email,
      firstname: 'Admin',
      lastname: 'User',
      name: 'Admin User',
      role: UserRole.ADMINISTRATOR,
      administrator: true,
      emailVerified: true,
    },
  });

  await prisma.account.upsert({
    where: { id: `${admin.id}-credential` },
    update: { password: hashedPassword },
    create: {
      id: `${admin.id}-credential`,
      userId: admin.id,
      accountId: admin.id,
      providerId: 'credential',
      password: hashedPassword,
    },
  });

  console.log('Admin user created:', admin.email);
}

seedAdmin()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
