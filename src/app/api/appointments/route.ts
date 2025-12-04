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
