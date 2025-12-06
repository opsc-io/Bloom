import { PrismaClient, UserRole } from '../src/generated/prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import { hashPassword } from 'better-auth/crypto';

const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is required');
}

const adapter = new PrismaPg({ connectionString });
const prisma = new PrismaClient({ adapter });

const PASSWORD = 'Password123!';

// Sample data
const therapists = [
  { firstname: 'Sarah', lastname: 'Johnson', email: 'therapist1@example.com' },
  { firstname: 'Michael', lastname: 'Chen', email: 'therapist2@example.com' },
  { firstname: 'Emily', lastname: 'Rodriguez', email: 'therapist3@example.com' },
  { firstname: 'David', lastname: 'Kim', email: 'therapist4@example.com' },
  { firstname: 'Jessica', lastname: 'Patel', email: 'therapist5@example.com' },
];

const patients = [
  { firstname: 'James', lastname: 'Wilson', email: 'patient1@example.com' },
  { firstname: 'Emma', lastname: 'Brown', email: 'patient2@example.com' },
  { firstname: 'Oliver', lastname: 'Davis', email: 'patient3@example.com' },
  { firstname: 'Sophia', lastname: 'Martinez', email: 'patient4@example.com' },
  { firstname: 'William', lastname: 'Garcia', email: 'patient5@example.com' },
];

async function createUser(
  data: { firstname: string; lastname: string; email: string },
  role: UserRole,
  hashedPassword: string
) {
  const id = crypto.randomUUID();
  const name = `${data.firstname} ${data.lastname}`;

  const user = await prisma.user.upsert({
    where: { email: data.email },
    update: {
      firstname: data.firstname,
      lastname: data.lastname,
      name,
      role,
      emailVerified: true,
    },
    create: {
      id,
      email: data.email,
      firstname: data.firstname,
      lastname: data.lastname,
      name,
      role,
      emailVerified: true,
    },
  });

  // Upsert credential account
  await prisma.account.upsert({
    where: { id: `${user.id}-credential` },
    update: { password: hashedPassword },
    create: {
      id: `${user.id}-credential`,
      userId: user.id,
      accountId: user.id,
      providerId: 'credential',
      password: hashedPassword,
    },
  });

  return user;
}

function daysFromNow(days: number, hour = 10, minutes = 0) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  d.setHours(hour, minutes, 0, 0);
  return d;
}

async function seed() {
  console.log('Seeding database...');

  // Hash password once (same for all test users)
  const hashedPassword = await hashPassword(PASSWORD);

  // Create therapists
  console.log('Creating therapists...');
  const createdTherapists = [];
  for (const t of therapists) {
    const user = await createUser(t, UserRole.THERAPIST, hashedPassword);
    createdTherapists.push(user);
    console.log(`  Created therapist: ${user.email}`);
  }

  // Create patients
  console.log('Creating patients...');
  const createdPatients = [];
  for (const p of patients) {
    const user = await createUser(p, UserRole.PATIENT, hashedPassword);
    createdPatients.push(user);
    console.log(`  Created patient: ${user.email}`);
  }

  // Create some appointments
  console.log('Creating appointments...');
  for (let i = 0; i < createdPatients.length; i++) {
    const patient = createdPatients[i];
    const therapist = createdTherapists[i % createdTherapists.length];

    // Upcoming appointment
    await prisma.appointment.create({
      data: {
        therapistId: therapist.id,
        patientId: patient.id,
        startAt: daysFromNow(i + 1, 9, 0),
        endAt: daysFromNow(i + 1, 9, 45),
        status: 'SCHEDULED',
      },
    });

    // Past completed appointment
    await prisma.appointment.create({
      data: {
        therapistId: therapist.id,
        patientId: patient.id,
        startAt: daysFromNow(-(i + 1), 14, 0),
        endAt: daysFromNow(-(i + 1), 14, 45),
        status: 'COMPLETED',
      },
    });

    console.log(`  Created appointments for ${patient.email} with ${therapist.email}`);
  }

  // Create some conversations
  console.log('Creating conversations...');
  for (let i = 0; i < 3; i++) {
    const patient = createdPatients[i];
    const therapist = createdTherapists[i];

    const conversation = await prisma.conversation.create({
      data: {
        participants: {
          create: [
            { userId: therapist.id, role: UserRole.THERAPIST },
            { userId: patient.id, role: UserRole.PATIENT },
          ],
        },
      },
    });

    // Add some messages
    const messages = [
      { senderId: therapist.id, body: 'Hi there! How are you feeling today?' },
      { senderId: patient.id, body: "I'm doing better, thanks for asking." },
      { senderId: therapist.id, body: 'Great to hear! Looking forward to our session.' },
    ];

    for (const msg of messages) {
      await prisma.message.create({
        data: {
          conversationId: conversation.id,
          senderId: msg.senderId,
          body: msg.body,
        },
      });
    }

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: { lastMessageAt: new Date() },
    });

    console.log(`  Created conversation between ${therapist.email} and ${patient.email}`);
  }

  console.log('\nSeed complete!');
  console.log('-------------------');
  console.log('Test credentials:');
  console.log(`  Password for all users: ${PASSWORD}`);
  console.log('\nTherapists:');
  createdTherapists.forEach((t) => console.log(`  - ${t.email}`));
  console.log('\nPatients:');
  createdPatients.forEach((p) => console.log(`  - ${p.email}`));
}

seed()
  .catch((err) => {
    console.error('Seeding failed:', err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
