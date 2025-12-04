import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import prisma from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { role } = body;

    if (!role || !["practitioner", "patient"].includes(role)) {
      return NextResponse.json(
        { error: "Invalid role. Must be 'practitioner' or 'patient'" },
        { status: 400 }
      );
    }

    // Update user role
    await prisma.user.update({
      where: { id: session.user.id },
      data: {
        therapist: role === "practitioner",
      },
    });

    return NextResponse.json({ success: true, role });
  } catch (error) {
    console.error("Error updating user role:", error);
    return NextResponse.json(
      { error: "Failed to update role" },
      { status: 500 }
    );
  }
}
