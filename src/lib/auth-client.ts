import { inferAdditionalFields } from "better-auth/client/plugins";
import { twoFactorClient } from "better-auth/client/plugins";
import { createAuthClient } from 'better-auth/react'
import type { auth } from "./auth";

const authClient = createAuthClient({
    plugins: [
        inferAdditionalFields<typeof auth>(),
        twoFactorClient({
            onTwoFactorRedirect() {
                window.location.href = "/verify-otp"
            }
        })
    ]
})

export const { signIn, signUp, signOut, useSession, twoFactor, sendVerificationEmail, $Infer } = authClient
