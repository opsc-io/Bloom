import { NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import prisma from "@/lib/prisma";

export async function GET() {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // For now, surface all therapists except the requester.
    const therapists = await prisma.user.findMany({
      where: {
        role: "THERAPIST",
        id: { not: session.user.id },
      },
      select: {
        id: true,
        firstname: true,
        lastname: true,
        email: true,
        image: true,
        role: true,
      },
      orderBy: { createdAt: "desc" },
      take: 100,
    });

    const mapped = therapists.map((t) => ({
      id: t.id,
      name: t.firstname || t.lastname ? `${t.firstname} ${t.lastname}`.trim() : t.email,
      email: t.email,
      image: t.image,
      role: t.role,
    }));

    return NextResponse.json({ therapists: mapped });
  } catch (err) {
    console.error("Failed to fetch available therapists", err);
    return NextResponse.json({ error: "Failed to fetch therapists" }, { status: 500 });
  }
}
