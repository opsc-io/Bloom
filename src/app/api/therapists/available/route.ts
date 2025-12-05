import { NextResponse } from "next/server";

export async function GET() {
  // TODO: Replace with actual database query
  // This should fetch all therapists who are accepting new patients
  const dummyTherapists = [
    {
      id: "therapist-1",
      name: "Dr. Sarah Johnson",
      email: "sarah.johnson@bloom.com",
      image: null,
      role: "THERAPIST",
    },
    {
      id: "therapist-2",
      name: "Dr. Michael Chen",
      email: "michael.chen@bloom.com",
      image: null,
      role: "THERAPIST",
    },
    {
      id: "therapist-3",
      name: "Dr. Emily Rodriguez",
      email: "emily.rodriguez@bloom.com",
      image: null,
      role: "THERAPIST",
    },
    {
      id: "therapist-4",
      name: "Dr. James Williams",
      email: "james.williams@bloom.com",
      image: null,
      role: "THERAPIST",
    },
    {
      id: "therapist-5",
      name: "Dr. Lisa Martinez",
      email: "lisa.martinez@bloom.com",
      image: null,
      role: "THERAPIST",
    },
  ];

  return NextResponse.json({ therapists: dummyTherapists });
}
