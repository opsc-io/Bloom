import { PrismaClient, UserRole } from '../src/generated/prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import { faker } from '@faker-js/faker';
import { hashPassword } from '../src/lib/password';

const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is required');
}

const adapter = new PrismaPg({ connectionString });
const prisma = new PrismaClient({ adapter });

const PASSWORD = 'Password123!';
const ADMIN_EMAIL = process.env.SEED_ADMIN_EMAIL || 'admin@bloomhealth.us';
const ADMIN_PASSWORD = process.env.SEED_ADMIN_PASSWORD || 'Admin1234!';
const THERAPIST_COUNT = 5;
const PATIENT_COUNT = 10;

type SeedUser = {
  firstname: string;
  lastname: string;
  email: string;
  bio?: string | null;
  image?: string | null;
};

const makeUserData = (): SeedUser => {
  const firstname = faker.person.firstName();
  const lastname = faker.person.lastName();
  return {
    firstname,
    lastname,
    email: faker.internet.email({ firstName: firstname, lastName: lastname }).toLowerCase(),
    bio: faker.lorem.sentence(),
    image: faker.image.avatar()
  };
};

const therapists = Array.from({ length: THERAPIST_COUNT }, () => makeUserData());
const patients = Array.from({ length: PATIENT_COUNT }, () => makeUserData());

async function createUser(data: SeedUser, role: UserRole, password: string) {
  const name = `${data.firstname} ${data.lastname}`;
  const hashedPassword = await hashPassword(password);
  const id = crypto.randomUUID();

  // Upsert user
  const user = await prisma.user.upsert({
    where: { email: data.email },
    update: {
      firstname: data.firstname,
      lastname: data.lastname,
      name,
      bio: data.bio ?? null,
      image: data.image ?? null,
      role,
      therapist: role === UserRole.THERAPIST,
      administrator: role === UserRole.ADMINISTRATOR,
      emailVerified: true,
    },
    create: {
      id,
      email: data.email,
      firstname: data.firstname,
      lastname: data.lastname,
      name,
      bio: data.bio ?? null,
      image: data.image ?? null,
      role,
      therapist: role === UserRole.THERAPIST,
      administrator: role === UserRole.ADMINISTRATOR,
      emailVerified: true,
    },
  });

  // Upsert credential account with password
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

async function seedFull() {
  console.log('Seeding database (full)...');

  // Seed admin
  console.log('Creating admin user...');
  await createUser(
    {
      firstname: 'Admin',
      lastname: 'User',
      email: ADMIN_EMAIL,
      bio: 'Administrator account',
      image: null,
    },
    UserRole.ADMINISTRATOR,
    ADMIN_PASSWORD
  );
  console.log(`  Admin user ready: ${ADMIN_EMAIL}`);

  // Create therapists
  console.log('Creating therapists...');
  const createdTherapists = [];
  for (const t of therapists) {
    const user = await createUser(t, UserRole.THERAPIST, PASSWORD);
    createdTherapists.push(user);
    console.log(`  Created therapist: ${user.email}`);
  }

  // Create patients
  console.log('Creating patients...');
  const createdPatients = [];
  for (const p of patients) {
    const user = await createUser(p, UserRole.PATIENT, PASSWORD);
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
  console.log(`  Admin password: ${ADMIN_PASSWORD}`);
  console.log('\nTherapists:');
  createdTherapists.forEach((t) => console.log(`  - ${t.email}`));
  console.log('\nPatients:');
  createdPatients.forEach((p) => console.log(`  - ${p.email}`));
}

seedFull()
  .catch((err) => {
    console.error('Seeding failed:', err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
