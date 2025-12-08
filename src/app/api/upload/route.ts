import { NextRequest, NextResponse } from "next/server";
import { Storage } from "@google-cloud/storage";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";

// Initialize GCS client - uses Application Default Credentials on GKE
const storage = new Storage();
const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "bloom-uploads";

export async function POST(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file || typeof file === "string") {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    const safeName = file.name?.replace(/\s+/g, "-").replace(/[^\w.\-]/g, "") || "upload";
    const key = `uploads/${session.user.id}/${Date.now()}-${safeName}`;

    const bucket = storage.bucket(BUCKET_NAME);
    const blob = bucket.file(key);

    const buffer = Buffer.from(await file.arrayBuffer());

    await blob.save(buffer, {
      contentType: file.type,
      metadata: {
        cacheControl: "public, max-age=31536000",
      },
    });

    // Make the file publicly accessible
    await blob.makePublic();

    const publicUrl = `https://storage.googleapis.com/${BUCKET_NAME}/${key}`;

    return NextResponse.json({
      url: publicUrl,
      pathname: key,
      contentType: file.type,
    });
  } catch (error) {
    console.error("Error uploading file:", error);
    return NextResponse.json(
      { error: "Upload failed" },
      { status: 500 }
    );
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { url } = await req.json();

    if (!url) {
      return NextResponse.json({ error: "URL is required" }, { status: 400 });
    }

    // Extract the file path from the GCS URL
    const urlObj = new URL(url);
    const pathname = urlObj.pathname.replace(`/${BUCKET_NAME}/`, "");

    // Only allow users to delete their own files
    if (!pathname.startsWith(`uploads/${session.user.id}/`)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const bucket = storage.bucket(BUCKET_NAME);
    await bucket.file(pathname).delete();

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting file:", error);
    return NextResponse.json(
      { error: "Delete failed" },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const searchParams = req.nextUrl.searchParams;
    const prefix = searchParams.get("prefix") || `uploads/${session.user.id}/`;

    // Only allow users to list their own files
    if (!prefix.startsWith(`uploads/${session.user.id}/`)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const bucket = storage.bucket(BUCKET_NAME);
    const [files] = await bucket.getFiles({ prefix });

    const blobs = files.map((file) => ({
      url: `https://storage.googleapis.com/${BUCKET_NAME}/${file.name}`,
      pathname: file.name,
      contentType: file.metadata.contentType || "application/octet-stream",
      contentDisposition: file.metadata.contentDisposition || "inline",
    }));

    return NextResponse.json({ blobs });
  } catch (error) {
    console.error("Error listing files:", error);
    return NextResponse.json(
      { error: "List failed" },
      { status: 500 }
    );
  }
}
