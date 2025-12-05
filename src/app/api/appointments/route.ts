import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/prisma";

const colors = ["bg-blue-500", "bg-purple-500", "bg-emerald-500", "bg-amber-500", "bg-pink-500", "bg-indigo-500"];

const pickColor = (id: string) => {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash << 5) - hash + id.charCodeAt(i);
  return colors[Math.abs(hash) % colors.length];
};

export async function GET(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userId = session.user.id;
  const isTherapist = session.user.role === "THERAPIST";

  const now = new Date();
  const startOfWeek = new Date(now);
  startOfWeek.setHours(0, 0, 0, 0);
  startOfWeek.setDate(startOfWeek.getDate() - ((startOfWeek.getDay() + 6) % 7)); // Monday
  const endOfWeek = new Date(startOfWeek);
  endOfWeek.setDate(endOfWeek.getDate() + 7);

  const appointments = await prisma.appointment.findMany({
    where: {
      startAt: { gte: startOfWeek, lt: endOfWeek },
      ...(isTherapist ? { therapistId: userId } : { patientId: userId }),
    },
    include: {
      patient: true,
      therapist: true,
    },
    orderBy: { startAt: "asc" },
  });

  const formatted = appointments.map((appt) => {
    const start = new Date(appt.startAt);
    const end = new Date(appt.endAt);
    const client = isTherapist
      ? `${appt.patient.firstname} ${appt.patient.lastname}`.trim() || appt.patient.email
      : `${appt.therapist.firstname} ${appt.therapist.lastname}`.trim() || appt.therapist.email;

    return {
      id: appt.id,
      title: appt.status === "SCHEDULED" ? "Appointment" : appt.status.toLowerCase(),
      start,
      end,
      durationMinutes: Math.max(15, Math.round((end.getTime() - start.getTime()) / 60000)),
      client,
      color: pickColor(appt.id),
      zoomLink: appt.zoomJoinUrl ?? undefined,
    };
  });

  return NextResponse.json({ appointments: formatted });
}

export async function POST(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { startAt, endAt, participantId } = body as {
    startAt?: string;
    endAt?: string;
    participantId?: string;
  };

  if (!startAt || !endAt || !participantId) {
    return NextResponse.json({ error: "startAt, endAt and participantId are required" }, { status: 400 });
  }

  const start = new Date(startAt);
  const end = new Date(endAt);
  if (isNaN(start.getTime()) || isNaN(end.getTime())) {
    return NextResponse.json({ error: "Invalid date format" }, { status: 400 });
  }
  if (end <= start) {
    return NextResponse.json({ error: "endAt must be after startAt" }, { status: 400 });
  }

  const isTherapist = session.user.role === "THERAPIST";
  const therapistId = isTherapist ? session.user.id : participantId;
  const patientId = isTherapist ? participantId : session.user.id;

  // Ensure participant exists
  const otherUser = await prisma.user.findUnique({
    where: { id: participantId },
    select: { id: true },
  });
  if (!otherUser) {
    return NextResponse.json({ error: "Participant not found" }, { status: 404 });
  }

  const appt = await prisma.appointment.create({
    data: {
      therapistId,
      patientId,
      startAt: start,
      endAt: end,
      status: "SCHEDULED",
    },
  });

  return NextResponse.json({ appointmentId: appt.id });
}

export async function PATCH(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { appointmentId, startAt, endAt } = body as { appointmentId?: string; startAt?: string; endAt?: string };
  if (!appointmentId || !startAt || !endAt) {
    return NextResponse.json({ error: "appointmentId, startAt and endAt are required" }, { status: 400 });
  }

  const appt = await prisma.appointment.findUnique({
    where: { id: appointmentId },
    select: { id: true, therapistId: true },
  });
  if (!appt) {
    return NextResponse.json({ error: "Appointment not found" }, { status: 404 });
  }

  if (appt.therapistId !== session.user.id) {
    return NextResponse.json({ error: "Only the therapist can edit this appointment" }, { status: 403 });
  }

  const start = new Date(startAt);
  const end = new Date(endAt);
  if (isNaN(start.getTime()) || isNaN(end.getTime()) || end <= start) {
    return NextResponse.json({ error: "Invalid start/end times" }, { status: 400 });
  }

  await prisma.appointment.update({
    where: { id: appointmentId },
    data: {
      startAt: start,
      endAt: end,
      status: "SCHEDULED",
    },
  });

  return NextResponse.json({ success: true });
}

export async function DELETE(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { appointmentId } = body as { appointmentId?: string };
  if (!appointmentId) {
    return NextResponse.json({ error: "appointmentId is required" }, { status: 400 });
  }

  const appt = await prisma.appointment.findUnique({
    where: { id: appointmentId },
    select: { id: true, patientId: true },
  });
  if (!appt) {
    return NextResponse.json({ error: "Appointment not found" }, { status: 404 });
  }

  if (appt.patientId !== session.user.id) {
    return NextResponse.json({ error: "Only the patient can cancel this appointment" }, { status: 403 });
  }

  await prisma.appointment.update({
    where: { id: appointmentId },
    data: { status: "CANCELLED" },
  });

  return NextResponse.json({ success: true });
}
