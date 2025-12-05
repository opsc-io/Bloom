import { faker } from "@faker-js/faker";
import prisma from "../src/lib/prisma";
import { auth } from "../src/lib/auth";

const capitalize = (str: string) => str.charAt(0).toUpperCase() + str.slice(1);
const seedPassword = "Password123!";
const THERAPIST_COUNT = 10;
const PATIENT_COUNT = 100;
const ADMIN_COUNT = 2;

async function clearData() {
  await prisma.$transaction([
    prisma.messageReaction.deleteMany(),
    prisma.messageAttachment.deleteMany(),
    prisma.message.deleteMany(),
    prisma.conversationParticipant.deleteMany(),
    prisma.conversation.deleteMany(),
    prisma.appointmentNote.deleteMany(),
    prisma.appointment.deleteMany(),
    prisma.account.deleteMany(),
    prisma.session.deleteMany(),
    prisma.verification.deleteMany(),
    prisma.user.deleteMany(),
  ]);
}

async function ensureUser({
  email,
  role,
  administrator = false,
  firstname,
  lastname,
}: {
  email: string;
  role: "THERAPIST" | "PATIENT" | "ADMINISTRATOR";
  administrator?: boolean;
  firstname?: string;
  lastname?: string;
}) {
  let user = await prisma.user.findUnique({ where: { email } });
  if (!user) {
    const first = firstname ?? capitalize(faker.person.firstName());
    const last = lastname ?? capitalize(faker.person.lastName());
    await auth.api.signUpEmail({
      body: {
        email,
        password: seedPassword,
        name: `${first} ${last}`,
        firstname: first,
        lastname: last,
        bio: faker.lorem.sentence(),
      },
    });
    user = await prisma.user.findUnique({ where: { email } });
  }

  if (!user) throw new Error(`Failed to create user ${email}`);

  if (user.role !== role || user.administrator !== administrator || user.emailVerified !== true) {
    user = await prisma.user.update({
      where: { id: user.id },
      data: {
        role,
        administrator,
        emailVerified: true,
      },
    });
  }

  return user;
}

async function seedUsers() {
  await clearData();

  const therapists = await Promise.all(
    Array.from({ length: THERAPIST_COUNT }, (_, idx) =>
      ensureUser({
        email: `therapist${idx + 1}@example.com`,
        role: "THERAPIST",
        firstname: capitalize(faker.person.firstName()),
        lastname: capitalize(faker.person.lastName()),
      })
    )
  );

  const patients = await Promise.all(
    Array.from({ length: PATIENT_COUNT }, (_, idx) =>
      ensureUser({
        email: `patient${idx + 1}@example.com`,
        role: "PATIENT",
        firstname: capitalize(faker.person.firstName()),
        lastname: capitalize(faker.person.lastName()),
      })
    )
  );

  const administrators = await Promise.all(
    Array.from({ length: ADMIN_COUNT }, (_, idx) =>
      ensureUser({
        email: `admin${idx + 1}@example.com`,
        role: "ADMINISTRATOR",
        administrator: true,
        firstname: capitalize(faker.person.firstName()),
        lastname: capitalize(faker.person.lastName()),
      })
    )
  );

  return { therapists, patients, administrators };
}

const pickTherapistForPatient = (patientIdx: number, therapists: Array<{ id: string }>) =>
  therapists[patientIdx % therapists.length];

async function seedAppointments({ therapists, patients }: Awaited<ReturnType<typeof seedUsers>>) {
  await prisma.appointmentNote.deleteMany();
  await prisma.appointment.deleteMany();

  const now = new Date();

  const appointments = await Promise.all(
    patients.map((patient, idx) => {
      const therapist = pickTherapistForPatient(idx, therapists);
      const startAt = new Date(now);
      startAt.setDate(startAt.getDate() + (idx % 14));
      startAt.setHours(10, 0, 0, 0);
      const endAt = new Date(startAt.getTime() + 60 * 60 * 1000);

      return prisma.appointment.create({
        data: {
          therapistId: therapist.id,
          patientId: patient.id,
          startAt,
          endAt,
          status: "SCHEDULED",
        },
      });
    })
  );

  return appointments;
}

async function seedConversations({ therapists, patients }: Awaited<ReturnType<typeof seedUsers>>) {
  await prisma.$transaction([
    prisma.messageReaction.deleteMany(),
    prisma.messageAttachment.deleteMany(),
    prisma.message.deleteMany(),
    prisma.conversationParticipant.deleteMany(),
    prisma.conversation.deleteMany(),
  ]);

  const conversations = [];

  for (let i = 0; i < patients.length; i++) {
    const patient = patients[i];
    const therapist = pickTherapistForPatient(i, therapists);
    const messageCount = faker.number.int({ min: 3, max: 8 });
    const messages = Array.from({ length: messageCount }, (_, idx) => ({
      body: faker.lorem.sentences(2),
      sender: { connect: { id: idx % 2 === 0 ? therapist.id : patient.id } },
      createdAt: faker.date.recent({ days: 10 }),
    })).sort((a, b) => (a.createdAt?.getTime() ?? 0) - (b.createdAt?.getTime() ?? 0));

    const convo = await prisma.conversation.create({
      data: {
        lastMessageAt: messages[messages.length - 1]?.createdAt ?? new Date(),
        participants: {
          create: [
            { role: "THERAPIST", user: { connect: { id: therapist.id } } },
            { role: "PATIENT", user: { connect: { id: patient.id } } },
          ],
        },
        messages: {
          create: messages,
        },
      },
    });

    conversations.push(convo);
  }

  return conversations;
}

async function main() {
  const users = await seedUsers();
  await seedAppointments(users);
  const conversations = await seedConversations(users);

  console.log("Seeded therapists:", users.therapists.length);
  console.log("Seeded patients:", users.patients.length);
  console.log("Seeded conversations:", conversations.length);
  console.log("Seeded admins:", users.administrators.map((a) => a.email));
  console.log("Seeded password for all users:", seedPassword);
}

main()
  .catch((error) => {
    console.error("Seeding failed:", error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
