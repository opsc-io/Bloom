"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { signIn, sendVerificationEmail } from "@/lib/auth-client";
import Image from "next/image";
import Link from "next/link";
import { LoginForm } from "@/components/login-form"
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Mail, CheckCircle } from "lucide-react";
import { useSession } from "@/lib/auth-client";


export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [emailNotVerified, setEmailNotVerified] = useState(false);
  const [unverifiedEmail, setUnverifiedEmail] = useState("");
  const [isResending, setIsResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && session?.user) {
      router.replace("/dashboard");
    }
  }, [isPending, session, router]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setEmailNotVerified(false);
    setIsSubmitting(true);

    const formData = new FormData(e.currentTarget);
    const email = formData.get("email") as string;

    try {
      const res = await signIn.email({
        email,
        password: formData.get("password") as string,
      });

      if (res.error) {
        // Check if it's an email verification error
        if (res.error.message?.toLowerCase().includes("email not verified") ||
          res.error.message?.toLowerCase().includes("verify your email")) {
          setEmailNotVerified(true);
          setUnverifiedEmail(email);
        } else {
          setError(res.error.message || "Something went wrong.");
        }
      } else {
        router.push("/dashboard");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleGoogleSignIn() {
    setError(null);
    setIsSubmitting(true);
    try {
      await signIn.social({ provider: "google" });
    } catch (err) {
      setError("Unable to start Google sign in. Please try again.");
      setIsSubmitting(false);
    }
  }

  async function handleResendVerification() {
    setIsResending(true);
    setResendSuccess(false);
    try {
      await sendVerificationEmail({
        email: unverifiedEmail,
      });
      setResendSuccess(true);
    } catch (err) {
      setError("Failed to resend verification email.");
    } finally {
      setIsResending(false);
    }
  }

  // Show email not verified message
  if (emailNotVerified) {
    return (
      <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
        <div className="flex w-full max-w-sm flex-col gap-6">
          <Link href="/" className="flex items-center gap-2 self-center font-medium">
            <Image src="/logo.svg" alt="Logo" width={150} height={150} />
          </Link>
          <Card>
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900/30">
                <Mail className="h-6 w-6 text-amber-600" />
              </div>
              <CardTitle className="text-xl">Email Not Verified</CardTitle>
              <CardDescription>
                Please verify your email before signing in
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground text-center">
                We sent a verification link to <strong>{unverifiedEmail}</strong>.
                Click the link in the email to verify your account.
              </p>

              {resendSuccess && (
                <div className="flex items-center gap-2 p-3 text-sm text-green-600 bg-green-50 dark:bg-green-900/20 rounded-md">
                  <CheckCircle className="h-4 w-4" />
                  Verification email resent!
                </div>
              )}

              {error && (
                <p className="text-sm text-red-500 text-center">{error}</p>
              )}

              <div className="space-y-2">
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={handleResendVerification}
                  disabled={isResending}
                >
                  {isResending ? "Sending..." : "Resend Verification Email"}
                </Button>
                <Button
                  variant="ghost"
                  className="w-full"
                  onClick={() => {
                    setEmailNotVerified(false);
                    setError(null);
                    setResendSuccess(false);
                  }}
                >
                  Try Different Email
                </Button>
              </div>

              <p className="text-xs text-muted-foreground text-center">
                Check your spam folder if you don&apos;t see the email.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (!isPending && session?.user) {
    return <p className="text-center mt-8 text-white">Redirecting...</p>;
  }

  return (
    <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <a href="#" className="flex items-center gap-2 self-center font-medium">
          <Image src="/logo.svg" alt="Logo" width={150} height={150} />
        </a>
        <LoginForm
          onSubmit={(e) => handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>)}
          onGoogleSignIn={handleGoogleSignIn}
          error={error}
          isSubmitting={isSubmitting}
        />

      </div>
    </div>
  );
}
