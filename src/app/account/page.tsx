"use client";

import { useState } from "react";
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
import { useSession, twoFactor } from "@/lib/auth-client";
import { Shield, ShieldCheck, ArrowLeft, Loader2 } from "lucide-react";

export default function AccountPage() {
  const router = useRouter();
  const { data: session, isPending, refetch } = useSession();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isEnabling, setIsEnabling] = useState(false);
  const [isDisabling, setIsDisabling] = useState(false);

  if (isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!session?.user) {
    router.push("/sign-in");
    return null;
  }

  const is2FAEnabled = (session.user as { twoFactorEnabled?: boolean }).twoFactorEnabled;

  async function handleEnable2FA() {
    if (!password) {
      setError("Please enter your password to enable 2FA");
      return;
    }

    setError(null);
    setSuccess(null);
    setIsEnabling(true);

    try {
      const result = await twoFactor.enable({ password });
      if (result.error) {
        setError(result.error.message || "Failed to enable 2FA");
      } else {
        setSuccess("Two-factor authentication has been enabled!");
        setPassword("");
        await refetch();
      }
    } catch (err) {
      setError("Failed to enable 2FA. Please try again.");
    } finally {
      setIsEnabling(false);
    }
  }

  async function handleDisable2FA() {
    if (!password) {
      setError("Please enter your password to disable 2FA");
      return;
    }

    setError(null);
    setSuccess(null);
    setIsDisabling(true);

    try {
      const result = await twoFactor.disable({ password });
      if (result.error) {
        setError(result.error.message || "Failed to disable 2FA");
      } else {
        setSuccess("Two-factor authentication has been disabled.");
        setPassword("");
        await refetch();
      }
    } catch (err) {
      setError("Failed to disable 2FA. Please try again.");
    } finally {
      setIsDisabling(false);
    }
  }

  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-6 bg-muted p-6 md:p-10">
      <div className="flex w-full max-w-md flex-col gap-6">
        <Link href="/dashboard" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <div className="flex items-center gap-2 self-center font-medium">
          <div className="bg-primary text-primary-foreground flex size-6 items-center justify-center rounded-md">
            <Image src="/logo.svg" alt="Logo" width={16} height={16} />
          </div>
          Bloom Health
        </div>

        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-xl">Account Settings</CardTitle>
            <CardDescription>
              Manage your account security settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* User Info */}
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Signed in as</p>
              <p className="font-medium">{session.user.email}</p>
            </div>

            {/* 2FA Section */}
            <div className="border-t pt-6">
              <div className="flex items-center gap-3 mb-4">
                {is2FAEnabled ? (
                  <ShieldCheck className="h-6 w-6 text-green-500" />
                ) : (
                  <Shield className="h-6 w-6 text-muted-foreground" />
                )}
                <div>
                  <h3 className="font-medium">Two-Factor Authentication</h3>
                  <p className="text-sm text-muted-foreground">
                    {is2FAEnabled
                      ? "2FA is enabled - your account is protected"
                      : "Add an extra layer of security to your account"}
                  </p>
                </div>
              </div>

              {error && (
                <div className="mb-4 p-3 text-sm text-red-600 bg-red-50 dark:bg-red-900/20 rounded-md">
                  {error}
                </div>
              )}

              {success && (
                <div className="mb-4 p-3 text-sm text-green-600 bg-green-50 dark:bg-green-900/20 rounded-md">
                  {success}
                </div>
              )}

              <div className="space-y-4">
                <div>
                  <label htmlFor="password" className="text-sm font-medium">
                    Enter your password to {is2FAEnabled ? "disable" : "enable"} 2FA
                  </label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="mt-2"
                  />
                </div>

                {is2FAEnabled ? (
                  <Button
                    variant="destructive"
                    onClick={handleDisable2FA}
                    disabled={isDisabling}
                    className="w-full"
                  >
                    {isDisabling ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Disabling...
                      </>
                    ) : (
                      "Disable Two-Factor Authentication"
                    )}
                  </Button>
                ) : (
                  <Button
                    onClick={handleEnable2FA}
                    disabled={isEnabling}
                    className="w-full"
                  >
                    {isEnabling ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Enabling...
                      </>
                    ) : (
                      "Enable Two-Factor Authentication"
                    )}
                  </Button>
                )}

                <p className="text-xs text-muted-foreground text-center">
                  When enabled, you'll receive a verification code via email each time you sign in.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
