// import { faker } from "@faker-js/faker";
// import prisma from "../src/lib/prisma";
// import { AppointmentStatus, UserRole, User } from "../src/generated/prisma/client";
// import { auth } from "../src/lib/auth";

// type SeedUser = {
//   id: string;
//   firstname: string;
//   lastname: string;
//   email: string;
//   role: UserRole;
// };

// const now = new Date();

// function daysFromNow(days: number, hour = 10, minutes = 0) {
//   const d = new Date(now);
//   d.setDate(d.getDate() + days);
//   d.setHours(hour, minutes, 0, 0);
//   return d;
// }

// async function clearData() {
//   await prisma.$transaction([
//     prisma.messageReaction.deleteMany(),
//     prisma.messageAttachment.deleteMany(),
//     prisma.message.deleteMany(),
//     prisma.conversationParticipant.deleteMany(),
//     prisma.conversation.deleteMany(),
//     prisma.appointmentNote.deleteMany(),
//     prisma.claim.deleteMany(),
//     prisma.appointment.deleteMany(),
//   ]);
// }

// async function ensureUser(u: SeedUser, password: string): Promise<User> {
//   try {
//     // Try to sign up via BetterAuth first (role is set afterwards)
//     const result = await auth.api.signUpEmail({
//       body: {
//         name: `${u.firstname} ${u.lastname}`.trim() || u.email,
//         email: u.email,
//         password,
//         firstname: u.firstname,
//         lastname: u.lastname,
//       } as Record<string, string>,
//     });

//     if (result?.user?.id) {
//       return prisma.user.update({
//         where: { id: result.user.id },
//         data: { emailVerified: true, role: u.role, name: `${u.firstname} ${u.lastname}`.trim() },
//       });
//     }

//     throw new Error("signUpEmail returned no user");
//   } catch (err) {
//     console.warn(`signUpEmail failed for ${u.email}, falling back to direct upsert:`, err);
//     return prisma.user.upsert({
//       where: { email: u.email },
//       update: {
//         firstname: u.firstname,
//         lastname: u.lastname,
//         role: u.role,
//         name: `${u.firstname} ${u.lastname}`.trim(),
//         emailVerified: true,
//       },
//       create: {
//         id: faker.string.uuid(),
//         firstname: u.firstname,
//         lastname: u.lastname,
//         email: u.email,
//         name: `${u.firstname} ${u.lastname}`.trim(),
//         role: u.role,
//         emailVerified: true,
//       },
//     });
//   }
// }

// async function seedUsers(): Promise<{ therapists: User[]; patients: User[] }> {
//   const therapists: SeedUser[] = Array.from({ length: 10 }).map((_, idx) => ({
//     id: faker.string.uuid(),
//     firstname: faker.person.firstName(),
//     lastname: faker.person.lastName(),
//     email: `therapist${idx + 1}@example.com`,
//     role: UserRole.THERAPIST,
//   }));

//   const patients: SeedUser[] = Array.from({ length: 10 }).map((_, idx) => ({
//     id: faker.string.uuid(),
//     firstname: faker.person.firstName(),
//     lastname: faker.person.lastName(),
//     email: `patient${idx + 1}@example.com`,
//     role: UserRole.PATIENT,
//   }));

//   const password = "Password123!";

//   // Clean old seed users to avoid unique conflicts
//   const seedEmails = [...therapists, ...patients].map((u) => u.email);
//   await prisma.user.deleteMany({ where: { email: { in: seedEmails } } });

//   const createdTherapists = await Promise.all(therapists.map((u) => ensureUser(u, password)));
//   const createdPatients = await Promise.all(patients.map((u) => ensureUser(u, password)));

//   return { therapists: createdTherapists, patients: createdPatients };
// }

// async function seedAppointments(therapistId: string, patientId: string) {
//   const appts = [
//     {
//       startAt: daysFromNow(1, 9, 0),
//       endAt: daysFromNow(1, 9, 45),
//       status: AppointmentStatus.SCHEDULED,
//     },
//     {
//       startAt: daysFromNow(3, 14, 0),
//       endAt: daysFromNow(3, 14, 45),
//       status: AppointmentStatus.SCHEDULED,
//     },
//     {
//       startAt: daysFromNow(-5, 11, 0),
//       endAt: daysFromNow(-5, 11, 50),
//       status: AppointmentStatus.COMPLETED,
//     },
//     {
//       startAt: daysFromNow(7, 16, 0),
//       endAt: daysFromNow(7, 16, 45),
//       status: AppointmentStatus.CANCELLED,
//     },
//   ];

//   for (const appt of appts) {
//     await prisma.appointment.create({
//       data: {
//         therapistId,
//         patientId,
//         startAt: appt.startAt,
//         endAt: appt.endAt,
//         status: appt.status,
//         zoomJoinUrl: "https://zoom.us/j/123456789",
//       },
//     });
//   }
// }

// async function seedConversations(therapistId: string, patientId: string) {
//   const conversation = await prisma.conversation.create({
//     data: {
//       participants: {
//         create: [
//           { userId: therapistId, role: UserRole.THERAPIST },
//           { userId: patientId, role: UserRole.PATIENT },
//         ],
//       },
//     },
//   });

//   const messages = [
//     { senderId: therapistId, body: "Hi there, thanks for reaching out. How are you feeling today?" },
//     { senderId: patientId, body: "I'm doing okay, looking forward to our session." },
//     { senderId: therapistId, body: "Great. Let's plan to review your goals tomorrow." },
//   ];

//   for (const msg of messages) {
//     await prisma.message.create({
//       data: {
//         conversationId: conversation.id,
//         senderId: msg.senderId,
//         body: msg.body,
//       },
//     });
//   }

//   await prisma.conversation.update({
//     where: { id: conversation.id },
//     data: { lastMessageAt: new Date() },
//   });

//   return conversation.id;
// }

// async function main() {
//   console.log("Clearing existing data...");
//   await clearData();

//   console.log("Seeding users...");
//   const { therapists, patients } = await seedUsers();

//   console.log("Seeding appointments and conversations...");
//   const convoIds: string[] = [];
//   for (const [idx, patient] of patients.entries()) {
//     const assignedTherapist = therapists[idx % therapists.length];
//     await seedAppointments(assignedTherapist.id, patient.id);
//     const convoId = await seedConversations(assignedTherapist.id, patient.id);
//     convoIds.push(convoId);

//     // Optional extra relationship: pair each patient with the next therapist for variety
//     const secondaryTherapist = therapists[(idx + 1) % therapists.length];
//     await seedAppointments(secondaryTherapist.id, patient.id);
//     const convoId2 = await seedConversations(secondaryTherapist.id, patient.id);
//     convoIds.push(convoId2);
//   }

//   console.log("Seed complete.");
//   console.log("Therapists:", therapists.map((t) => t.email));
//   console.log("Patients:", patients.map((p) => p.email));
//   console.log("Conversations:", convoIds);
// }

// main()
//   .catch((err) => {
//     console.error("Seeding failed:", err);
//     process.exit(1);
//   })
//   .finally(async () => {
//     await prisma.$disconnect();
//   });
