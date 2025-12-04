import { auth } from "@/lib/auth";
import { toNextJsHandler } from "better-auth/next-js";

// Force Node.js runtime for nodemailer
export const runtime = 'nodejs';

export const { POST, GET } = toNextJsHandler(auth);