"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { signIn, signUp, sendVerificationEmail } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { SignupForm } from "@/components/signup-form"
import { Mail, CheckCircle } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

export default function SignUpPage() {
    const router = useRouter();
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [signupSuccess, setSignupSuccess] = useState(false);
    const [userEmail, setUserEmail] = useState("");
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
        setIsSubmitting(true);

        const formData = new FormData(e.currentTarget);
        const password = (formData.get("password") as string) || "";
        const confirmPassword = (formData.get("confirmPassword") as string) || "";
        const email = (formData.get("email") as string) || "";

        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            setIsSubmitting(false);
            return;
        }

        const res = await (signUp.email as (params: {
            name: string;
            email: string;
            password: string;
            firstname?: string;
            lastname?: string;
        }) => Promise<{ error?: { message?: string } }>)({
            name: `${(formData.get("firstName") as string) || ""} ${(formData.get("lastName") as string) || ""}`.trim(),
            email,
            password,
            firstname: (formData.get("firstName") as string) || "",
            lastname: (formData.get("lastName") as string) || "",
        });

        if (res.error) {
            setError(res.error.message || "Something went wrong.");
        } else {
            setUserEmail(email);
            setSignupSuccess(true);
        }

        setIsSubmitting(false);
    }

    async function handleGoogleSignup() {
        setError(null);
        setIsSubmitting(true);
        try {
            await signIn.social({ provider: "google" });
        } catch (err) {
            setError("Unable to start Google sign up. Please try again.");
            setIsSubmitting(false);
        }
    }

    async function handleResendVerification() {
        setIsResending(true);
        setResendSuccess(false);
        try {
            await sendVerificationEmail({
                email: userEmail,
            });
            setResendSuccess(true);
        } catch (err) {
            setError("Failed to resend verification email.");
        } finally {
            setIsResending(false);
        }
    }

    if (!isPending && session?.user) {
        return <p className="text-center mt-8 text-white">Redirecting...</p>;
    }

    // Show success message after signup
    if (signupSuccess) {
        return (
            <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
                <div className="flex w-full max-w-sm flex-col gap-6">
                    <Link href="/" className="flex items-center gap-8 self-center font-medium">
                        <Image
                            src="/logo.svg"
                            alt="Bloom logo"
                            width={175}
                            height={175}
                        />
                    </Link>
                    <Card>
                        <CardHeader className="text-center">
                            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                                <Mail className="h-6 w-6 text-primary" />
                            </div>
                            <CardTitle className="text-xl">Check Your Email</CardTitle>
                            <CardDescription>
                                We sent a verification link to <strong>{userEmail}</strong>
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <p className="text-sm text-muted-foreground text-center">
                                Click the link in the email to verify your account and start using Bloom Health.
                            </p>

                            {resendSuccess && (
                                <div className="flex items-center gap-2 p-3 text-sm text-green-600 bg-green-50 dark:bg-green-900/20 rounded-md">
                                    <CheckCircle className="h-4 w-4" />
                                    Verification email resent!
                                </div>
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
                                <Button asChild variant="ghost" className="w-full">
                                    <Link href="/sign-in">Back to Sign In</Link>
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

    return (
        <div className="bg-muted flex min-h-svh flex-col items-center justify-center gap-6 p-6 md:p-10">
            <div className="flex w-full max-w-sm flex-col gap-6">
                <a href="#" className="flex items-center gap-8 self-center font-medium">
                    <Image
                        src="/logo.svg"
                        alt="Bloom logo"
                        width={175}
                        height={175}
                    />
                </a>
                <SignupForm
                    onSubmit={(e: React.FormEvent) => { void handleSubmit(e as React.FormEvent<HTMLFormElement>); }}
                    error={error}
                    isSubmitting={isSubmitting}
                    onGoogleSignUp={handleGoogleSignup}
                />
            </div>
        </div>
    )
}
