import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import prisma from "@/lib/prisma";

export async function GET() {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = session.user.id;
    const userRole = (session.user as { role?: string }).role;

    const people: Array<{
      id: string;
      name: string;
      email: string;
      image?: string | null;
      role?: string;
    }> = [];

    if (userRole === "THERAPIST") {
      const appointments = await prisma.appointment.findMany({
        where: { therapistId: userId },
        select: {
          patient: {
            select: {
              id: true,
              firstname: true,
              lastname: true,
              email: true,
              image: true,
              role: true,
            },
          },
        },
        orderBy: { startAt: "desc" },
        take: 50,
      });

      const seen = new Set<string>();
      for (const appt of appointments) {
        const patient = appt.patient;
        if (patient && !seen.has(patient.id)) {
          seen.add(patient.id);
          people.push({
            id: patient.id,
            name:
              patient.firstname || patient.lastname
                ? `${patient.firstname} ${patient.lastname}`.trim()
                : patient.email,
            email: patient.email,
            image: patient.image,
            role: patient.role,
          });
        }
      }
    } else if (userRole === "PATIENT") {
      const appointments = await prisma.appointment.findMany({
        where: { patientId: userId },
        select: {
          therapist: {
            select: {
              id: true,
              firstname: true,
              lastname: true,
              email: true,
              image: true,
              role: true,
            },
          },
        },
        orderBy: { startAt: "desc" },
        take: 5,
      });

      const seen = new Set<string>();
      for (const appt of appointments) {
        const therapist = appt.therapist;
        if (therapist && !seen.has(therapist.id)) {
          seen.add(therapist.id);
          people.push({
            id: therapist.id,
            name:
              therapist.firstname || therapist.lastname
                ? `${therapist.firstname} ${therapist.lastname}`.trim()
                : therapist.email,
            email: therapist.email,
            image: therapist.image,
            role: therapist.role,
          });
        }
      }
    }

    return NextResponse.json({ people });
  } catch (error) {
    console.error("Error fetching connections:", error);
    return NextResponse.json(
      { error: "Failed to fetch connections" },
      { status: 500 }
    );
  }
}
