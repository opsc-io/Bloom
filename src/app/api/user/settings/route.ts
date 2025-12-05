import { NextRequest, NextResponse } from "next/server";
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

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: {
        firstname: true,
        lastname: true,
        email: true,
        bio: true,
        image: true,
        accounts: {
          select: {
            providerId: true,
          },
        },
      },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const allowPasswordChange = user.accounts.some(
      (acct) => acct.providerId === "email" || acct.providerId === "credentials"
    );

    return NextResponse.json({
      user: {
        firstname: user.firstname,
        lastname: user.lastname,
        email: user.email,
        bio: user.bio,
        image: user.image,
      },
      accounts: user.accounts,
      allowPasswordChange,
    });
  } catch (error) {
    console.error("Error fetching profile:", error);
    return NextResponse.json(
      { error: "Failed to load profile" },
      { status: 500 }
    );
  }
}

export async function PATCH(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { firstname, lastname, bio, image } = body;

    if (
      firstname === undefined &&
      lastname === undefined &&
      bio === undefined &&
      image === undefined
    ) {
      return NextResponse.json(
        { error: "No fields provided to update" },
        { status: 400 }
      );
    }

    // Update user profile
    const updatedUser = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        ...(firstname !== undefined && { firstname }),
        ...(lastname !== undefined && { lastname }),
        ...(bio !== undefined && { bio }),
        ...(image !== undefined && { image }),
      },
    });

    return NextResponse.json({ 
      success: true, 
      user: {
        firstname: updatedUser.firstname,
        lastname: updatedUser.lastname,
        bio: updatedUser.bio,
        image: updatedUser.image,
      }
    });
  } catch (error) {
    console.error("Error updating profile:", error);
    return NextResponse.json(
      { error: "Failed to update profile" },
      { status: 500 }
    );
  }
}
