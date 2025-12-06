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
      // For therapists: show all patients they can book appointments with
      const patients = await prisma.user.findMany({
        where: {
          role: "PATIENT",
          id: { not: userId },
        },
        select: {
          id: true,
          firstname: true,
          lastname: true,
          email: true,
          image: true,
          role: true,
        },
        take: 100,
      });
      patients.forEach((p) => {
        people.push({
          id: p.id,
          name: p.firstname || p.lastname ? `${p.firstname} ${p.lastname}`.trim() : p.email,
          email: p.email,
          image: p.image,
          role: p.role ?? undefined,
        });
      });
    } else if (userRole === "PATIENT") {
      // For patients: show all therapists they can book appointments with
      const therapists = await prisma.user.findMany({
        where: {
          role: "THERAPIST",
          id: { not: userId },
        },
        select: {
          id: true,
          firstname: true,
          lastname: true,
          email: true,
          image: true,
          role: true,
        },
        take: 100,
      });
      therapists.forEach((t) => {
        people.push({
          id: t.id,
          name: t.firstname || t.lastname ? `${t.firstname} ${t.lastname}`.trim() : t.email,
          email: t.email,
          image: t.image,
          role: t.role ?? undefined,
        });
      });
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
