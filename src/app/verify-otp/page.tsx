"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { twoFactor } from "@/lib/auth-client";
import { Shield, Loader2, Mail } from "lucide-react";

export default function VerifyOTPPage() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [codeSent, setCodeSent] = useState(false);

  // Send OTP on page load
  useEffect(() => {
    sendOTP();
  }, []);

  async function sendOTP() {
    setIsSending(true);
    setError(null);

    try {
      const result = await twoFactor.sendOtp();
      if (result.error) {
        setError(result.error.message || "Failed to send verification code");
      } else {
        setCodeSent(true);
      }
    } catch (err) {
      setError("Failed to send verification code. Please try again.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleVerify() {
    if (!code || code.length < 6) {
      setError("Please enter a valid 6-digit code");
      return;
    }

    setError(null);
    setIsVerifying(true);

    try {
      const result = await twoFactor.verifyOtp({
        code,
        trustDevice: true, // Trust this device for 30 days
      });

      if (result.error) {
        setError(result.error.message || "Invalid verification code");
      } else {
        router.push("/dashboard");
      }
    } catch (err) {
      setError("Failed to verify code. Please try again.");
    } finally {
      setIsVerifying(false);
    }
  }

  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-6 bg-muted p-6 md:p-10">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <div className="flex items-center gap-2 self-center font-medium">
          <div className="bg-primary text-primary-foreground flex size-6 items-center justify-center rounded-md">
            <Image src="/logo.svg" alt="Logo" width={16} height={16} />
          </div>
          Bloom Health
        </div>

        <Card>
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <CardTitle className="text-xl">Verify Your Identity</CardTitle>
            <CardDescription>
              Enter the verification code sent to your email
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {codeSent && (
              <div className="flex items-center gap-2 p-3 text-sm text-green-600 bg-green-50 dark:bg-green-900/20 rounded-md">
                <Mail className="h-4 w-4" />
                Verification code sent to your email
              </div>
            )}

            {error && (
              <div className="p-3 text-sm text-red-600 bg-red-50 dark:bg-red-900/20 rounded-md">
                {error}
              </div>
            )}

            <div className="space-y-2">
              <label htmlFor="code" className="text-sm font-medium">
                Verification Code
              </label>
              <Input
                id="code"
                type="text"
                inputMode="numeric"
                placeholder="Enter 6-digit code"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                className="text-center text-2xl tracking-widest"
                maxLength={6}
              />
            </div>

            <Button
              onClick={handleVerify}
              disabled={isVerifying || code.length < 6}
              className="w-full"
            >
              {isVerifying ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Verifying...
                </>
              ) : (
                "Verify"
              )}
            </Button>

            <div className="text-center">
              <Button
                variant="link"
                onClick={sendOTP}
                disabled={isSending}
                className="text-sm"
              >
                {isSending ? "Sending..." : "Resend code"}
              </Button>
            </div>

            <div className="text-center text-sm text-muted-foreground">
              <Link href="/sign-in" className="underline hover:text-foreground">
                Back to sign in
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
