"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { signUp } from "@/lib/auth-client";

import { SignupForm } from "@/components/signup-form"

export default function SignUpPage() {
    const router = useRouter();
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setError(null);
        setIsSubmitting(true);

        const formData = new FormData(e.currentTarget);
        const password = (formData.get("password") as string) || "";
        const confirmPassword = (formData.get("confirmPassword") as string) || "";

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
            email: (formData.get("email") as string) || "",
            password,
            firstname: (formData.get("firstName") as string) || "",
            lastname: (formData.get("lastName") as string) || "",
        });

        if (res.error) {
            setError(res.error.message || "Something went wrong.");
        } else {
            router.push("/dashboard");
        }

        setIsSubmitting(false);
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
                />
            </div>
        </div>
    )
}
