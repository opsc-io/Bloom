import { generateRegistrationOptions, verifyRegistrationResponse, generateAuthenticationOptions, verifyAuthenticationResponse } from '@simplewebauthn/server';
import { RP_ID, ORIGIN } from '../config';
import { storeChallenge, getChallenge, clearChallenge, storeCredential, updateCredentialCounter, getUserCredential } from '../services/webauthnStorage';

// Human-readable titles for your website
export const relyingParty = {
  name: 'Therapy Practice Platform',
  id: RP_ID,
};

/**
 * Initialize a WebAuthn registration for a user
 * @param user - The user initiating registration
 * @returns Registration options for the client
 */
export async function initRegistration(user: any) {
  const options = await generateRegistrationOptions({
    rpName: relyingParty.name,
    rpID: relyingParty.id,
    userID: user.user_id,
    userName: user.email,
    // Don't prompt users for additional information about the authenticator
    // (Recommended for smoother UX)
    attestationType: 'none',
    // Prevent users from re-registering existing authenticators
    excludeCredentials: user.passkey_credential ? [
      {
        id: user.passkey_credential.id,
        transports: user.passkey_credential.transports,
      },
    ] : [],
  });

  // Remember the challenge for later verification
  // In production, store this in a database or cache
  await storeChallenge(user.user_id, (options as any).challenge);

  return options;
}

/**
 * Verify WebAuthn registration response
 * @param user - The user completing registration
 * @param response - The response from the authenticator
 * @returns Verification result
 */
export async function verifyRegistration(user: any, response: any) {
  const expectedChallenge = await getChallenge(user.user_id);
  
  if (!expectedChallenge) {
    throw new Error('Registration challenge not found');
  }

  try {
    const verification = await verifyRegistrationResponse({
      response,
      expectedChallenge,
  expectedOrigin: ORIGIN,
      expectedRPID: relyingParty.id,
    });

    const { verified, registrationInfo } = verification as any;

    if (verified && registrationInfo) {
      // registrationInfo shapes may vary between versions; normalize via `any`
      const credentialID = registrationInfo.credentialID ?? registrationInfo.credential?.id;
      const credentialPublicKey = registrationInfo.credentialPublicKey ?? registrationInfo.credential?.publicKey;
      const counter = registrationInfo.counter ?? registrationInfo.credential?.counter;

      // Store the credential information
      await storeCredential(user.user_id, {
        id: credentialID,
        publicKey: credentialPublicKey,
        counter,
        transports: response.response.transports,
      });

      // Clear the challenge
      await clearChallenge(user.user_id);
    }

    return verified;
  } catch (error) {
    console.error('WebAuthn registration verification failed:', error);
    return false;
  }
}

/**
 * Initialize a WebAuthn authentication for a user
 * @param user - The user initiating authentication
 * @returns Authentication options for the client
 */
export async function initAuthentication(user: any) {
  if (!user.passkey_credential) {
    throw new Error('User has no registered passkey');
  }

  const options = await generateAuthenticationOptions({
    rpID: relyingParty.id,
    // Require users to use a previously registered authenticator
    allowCredentials: [
      {
        id: user.passkey_credential.id,
        transports: user.passkey_credential.transports,
      },
    ],
    // Prevent users from re-authenticating with the same authenticator
    userVerification: 'preferred',
  });

  // Remember the challenge for later verification
  await storeChallenge(user.user_id, (options as any).challenge);

  return options;
}

/**
 * Verify WebAuthn authentication response
 * @param user - The user completing authentication
 * @param response - The response from the authenticator
 * @returns Verification result
 */
export async function verifyAuthentication(user: any, response: any) {
  const expectedChallenge = await getChallenge(user.user_id);
  
  if (!expectedChallenge) {
    throw new Error('Authentication challenge not found');
  }

  if (!user.passkey_credential) {
    throw new Error('User has no registered passkey');
  }

  try {
    // Cast to any because typings differ across versions; ensure runtime values are correct
    const verification = await verifyAuthenticationResponse({
      response,
      expectedChallenge,
  expectedOrigin: ORIGIN,
      expectedRPID: relyingParty.id,
      // The verifier needs authenticator info to validate the counter/public key
      // Shape can vary; pass a permissive any and let runtime validation occur.
      authenticator: {
        credentialID: user.passkey_credential.id,
        credentialPublicKey: user.passkey_credential.publicKey,
        counter: user.passkey_credential.counter,
      } as any,
    } as any);

    const { verified, authenticationInfo } = verification as any;

    if (verified) {
      // Update the counter to prevent replay attacks
      await updateCredentialCounter(user.user_id, authenticationInfo?.newCounter ?? authenticationInfo?.counter);

      // Clear the challenge
      await clearChallenge(user.user_id);
    }

    return verified;
  } catch (error) {
    console.error('WebAuthn authentication verification failed:', error);
    return false;
  }
}

// Storage functions are implemented in `src/services/webauthnStorage.ts`
