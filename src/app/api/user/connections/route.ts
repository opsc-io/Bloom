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
      // all distinct patients with appointments with this therapist
      const patientAppointments = await prisma.appointment.findMany({
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
        take: 500,
      });
      const seen = new Set<string>();
      patientAppointments.forEach((appt) => {
        const p = appt.patient;
        if (p && !seen.has(p.id)) {
          seen.add(p.id);
          people.push({
            id: p.id,
            name: p.firstname || p.lastname ? `${p.firstname} ${p.lastname}`.trim() : p.email,
            email: p.email,
            image: p.image,
            role: p.role,
          });
        }
      });
      // If no patients yet, allow selection from the patient pool to create first appointment
      if (people.length === 0) {
        const patients = await prisma.user.findMany({
          where: { role: "PATIENT" },
          select: {
            id: true,
            firstname: true,
            lastname: true,
            email: true,
            image: true,
            role: true,
          },
          orderBy: { createdAt: "desc" },
          take: 200,
        });
        patients.forEach((p) => {
          people.push({
            id: p.id,
            name: p.firstname || p.lastname ? `${p.firstname} ${p.lastname}`.trim() : p.email,
            email: p.email,
            image: p.image,
            role: p.role,
          });
        });
      }
    } else if (userRole === "PATIENT") {
      // all distinct therapists this patient has seen or has appointments with
      const therapistAppointments = await prisma.appointment.findMany({
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
        take: 200,
      });
      const seen = new Set<string>();
      therapistAppointments.forEach((appt) => {
        const t = appt.therapist;
        if (t && !seen.has(t.id)) {
          seen.add(t.id);
          people.push({
            id: t.id,
            name: t.firstname || t.lastname ? `${t.firstname} ${t.lastname}`.trim() : t.email,
            email: t.email,
            image: t.image,
            role: t.role,
          });
        }
      });
      // If no prior therapists, allow selection from therapist pool to create first appointment
      if (people.length === 0) {
        const therapists = await prisma.user.findMany({
          where: { role: "THERAPIST" },
          select: {
            id: true,
            firstname: true,
            lastname: true,
            email: true,
            image: true,
            role: true,
          },
          orderBy: { createdAt: "desc" },
          take: 200,
        });
        therapists.forEach((t) => {
          people.push({
            id: t.id,
            name: t.firstname || t.lastname ? `${t.firstname} ${t.lastname}`.trim() : t.email,
            email: t.email,
            image: t.image,
            role: t.role,
          });
        });
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
