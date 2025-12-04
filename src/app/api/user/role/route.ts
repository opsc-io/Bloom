import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { auth } from "@/lib/auth";

type RolePayload = { role?: "patient" | "practitioner" };
type UserRole = "ADMINISTRATOR" | "THERAPIST" | "PATIENT" | "UNSET";

const deriveRole = (role: UserRole | null | undefined) =>
  !role || role === "UNSET"
    ? null
    : role === "THERAPIST"
      ? "practitioner"
      : role === "PATIENT"
        ? "patient"
        : "administrator";

export async function GET(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const role = deriveRole(session.user.role as UserRole);
  const needsSelection = !role;
  return NextResponse.json({ ok: true, role, needsSelection, flags: { role: session.user.role } });
}

export async function POST(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  let body: RolePayload;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  if (body.role !== "patient" && body.role !== "practitioner") {
    return NextResponse.json({ error: "Invalid role" }, { status: 400 });
  }

  const updated = await prisma.user.update({
    where: { id: session.user.id },
    data: { role: body.role === "practitioner" ? "THERAPIST" : "PATIENT" },
    select: { role: true },
  });

  const role = deriveRole(updated.role as UserRole);
  return NextResponse.json({
    ok: true,
    role,
    needsSelection: !role,
    flags: updated,
  });
}
