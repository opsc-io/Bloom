"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
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
import { CheckCircle, XCircle, Loader2, Mail } from "lucide-react";

function VerifyEmailContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const token = searchParams.get("token");

    const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!token) {
            return;
        }

        let cancelled = false;
        async function verifyEmail() {
            try {
                const res = await fetch(`/api/auth/verify-email?token=${token}`, {
                    method: "GET",
                });

                if (res.ok) {
                    if (cancelled) return;
                    setStatus("success");
                    setTimeout(() => {
                        router.push("/sign-in");
                    }, 3000);
                } else {
                    const data = await res.json();
                    if (cancelled) return;
                    setStatus("error");
                    setError(data.error?.message || data.message || "Verification failed");
                }
            } catch (err) {
                if (cancelled) return;
                setStatus("error");
                setError("Failed to verify email. Please try again.");
            }
        }

        verifyEmail();
        return () => {
            cancelled = true;
        };
    }, [token, router]);

    return (
        <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
            <div className="flex w-full max-w-sm flex-col gap-6">
                <Link href="/" className="flex items-center gap-2 self-center font-medium">
                    <Image src="/logo.svg" alt="Logo" width={150} height={150} />
                </Link>

                <Card>
                    <CardHeader className="text-center">
                        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full">
                            {status === "loading" && (
                                <div className="bg-muted">
                                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                </div>
                            )}
                            {status === "success" && (
                                <div className="bg-green-100 dark:bg-green-900/30 rounded-full p-3">
                                    <CheckCircle className="h-6 w-6 text-green-600" />
                                </div>
                            )}
                            {status === "error" && (
                                <div className="bg-red-100 dark:bg-red-900/30 rounded-full p-3">
                                    <XCircle className="h-6 w-6 text-red-600" />
                                </div>
                            )}
                        </div>
                        <CardTitle className="text-xl">
                            {status === "loading" && "Verifying Email..."}
                            {status === "success" && "Email Verified!"}
                            {status === "error" && "Verification Failed"}
                        </CardTitle>
                        <CardDescription>
                            {status === "loading" && "Please wait while we verify your email address"}
                            {status === "success" && "Your email has been successfully verified"}
                            {status === "error" && (error || "Something went wrong")}
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {status === "success" && (
                            <div className="space-y-4">
                                <p className="text-sm text-muted-foreground text-center">
                                    Redirecting you to sign in...
                                </p>
                                <Button asChild className="w-full">
                                    <Link href="/sign-in">Sign In Now</Link>
                                </Button>
                            </div>
                        )}
                        {status === "error" && (
                            <div className="space-y-4">
                                <p className="text-sm text-muted-foreground text-center">
                                    The verification link may have expired or is invalid.
                                </p>
                                <Button asChild className="w-full">
                                    <Link href="/sign-in">Back to Sign In</Link>
                                </Button>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}

function LoadingFallback() {
    return (
        <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
            <div className="flex w-full max-w-sm flex-col gap-6">
                <div className="flex items-center justify-center">
                    <Image src="/logo.svg" alt="Logo" width={150} height={150} />
                </div>
                <Card>
                    <CardHeader className="text-center">
                        <div className="mx-auto mb-4">
                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        </div>
                        <CardTitle className="text-xl">Verify Email</CardTitle>
                        <CardDescription>Loading...</CardDescription>
                    </CardHeader>
                </Card>
            </div>
        </div>
    );
}

export default function VerifyEmailPage() {
    return (
        <Suspense fallback={<LoadingFallback />}>
            <VerifyEmailContent />
        </Suspense>
    );
}
