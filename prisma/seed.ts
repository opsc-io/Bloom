import { faker } from "@faker-js/faker";
import prisma from "../src/lib/prisma";

const capitalize = (str: string) => str.charAt(0).toUpperCase() + str.slice(1);

async function seedUsers() {
  const therapist = await prisma.user.upsert({
    where: { email: "therapist@example.com" },
    update: {},
    create: {
      id: faker.string.uuid(),
      firstname: capitalize(faker.person.firstName()),
      lastname: capitalize(faker.person.lastName()),
      name: "Therapist Seed",
      email: "therapist@example.com",
      role: "THERAPIST",
    },
  });

  const patients = await Promise.all(
    Array.from({ length: 3 }, (_, idx) =>
      prisma.user.upsert({
        where: { email: `patient${idx + 1}@example.com` },
        update: {},
        create: {
          id: faker.string.uuid(),
          firstname: capitalize(faker.person.firstName()),
          lastname: capitalize(faker.person.lastName()),
          name: `Patient ${idx + 1}`,
          email: `patient${idx + 1}@example.com`,
          role: "PATIENT",
        },
      })
    )
  );

  return { therapist, patients };
}

async function seedAppointments({ therapist, patients }: Awaited<ReturnType<typeof seedUsers>>) {
  await prisma.appointmentNote.deleteMany();
  await prisma.appointment.deleteMany();

  const now = new Date();

  const appointments = await Promise.all(
    patients.map((patient, idx) => {
      const startAt = new Date(now);
      startAt.setDate(startAt.getDate() + idx);
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

async function seedConversations({ therapist, patients }: Awaited<ReturnType<typeof seedUsers>>) {
  await prisma.$transaction([
    prisma.messageReaction.deleteMany(),
    prisma.messageAttachment.deleteMany(),
    prisma.message.deleteMany(),
    prisma.conversationParticipant.deleteMany(),
    prisma.conversation.deleteMany(),
  ]);

  const conversations = [];

  for (const patient of patients) {
    const messageCount = faker.number.int({ min: 3, max: 6 });
    const messages = Array.from({ length: messageCount }, (_, idx) => ({
      body: faker.lorem.sentences(2),
      sender: { connect: { id: idx % 2 === 0 ? therapist.id : patient.id } },
      createdAt: faker.date.recent({ days: 5 }),
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

  console.log("Seeded therapist:", users.therapist.email);
  console.log("Seeded patients:", users.patients.map((p) => p.email));
  console.log("Seeded conversations:", conversations.map((c) => c.id));
}

main()
  .catch((error) => {
    console.error("Seeding failed:", error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
