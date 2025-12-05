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

    // Dummy data for testing - will be replaced with actual database relationships
    const people: Array<{
      id: string;
      name: string;
      email: string;
      image?: string | null;
      role?: string;
    }> = userRole === "THERAPIST" ? [
      {
        id: "1",
        name: "Sarah Johnson",
        email: "sarah.j@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "2",
        name: "Michael Chen",
        email: "m.chen@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "3",
        name: "Emily Rodriguez",
        email: "emily.r@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "4",
        name: "David Kim",
        email: "david.kim@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "5",
        name: "Jessica Martinez",
        email: "j.martinez@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "6",
        name: "Robert Taylor",
        email: "robert.t@example.com",
        image: null,
        role: "PATIENT"
      },
      {
        id: "7",
        name: "Amanda Wilson",
        email: "amanda.w@example.com",
        image: null,
        role: "PATIENT"
      }
    ] : [
      {
        id: "therapist-1",
        name: "Dr. Jennifer Smith",
        email: "dr.smith@bloom.com",
        image: null,
        role: "THERAPIST"
      }
    ];

    // TODO: Implement actual patient-therapist relationship lookup
    // If THERAPIST role: fetch all patients assigned to this therapist
    // If PATIENT role: fetch the assigned therapist

    return NextResponse.json({ people });
  } catch (error) {
    console.error("Error fetching connections:", error);
    return NextResponse.json(
      { error: "Failed to fetch connections" },
      { status: 500 }
    );
  }
}
