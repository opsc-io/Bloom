<div align="center">
  <!-- place `public/mess.png` to show the app logo here -->
  <img src="/mess.png" alt="App logo" width="160" />
  <h1>OPSC — Therapy Practice Platform</h1>
</div>

Overview
--------
OPSC helps therapists set up and operate a practice from onboarding to billing and patient care. It combines credentialing automation, telehealth, payments, a lightweight EHR, and realtime messaging in two delivery modes: self-hosted and cloud multi-tenant.

Product vision
--------------
Create an open-source platform inspired by Alma and Headway, offering:

- A self-hosted guided workflow for therapists who want full control.
- A low-cost managed multi-tenant SaaS for therapists who prefer a hosted option.
<div align="center">
  <!-- place `public/mess.png` to show the app logo here -->
  <img src="/mess.png" alt="App logo" width="160" />
  <h1>OPSC — Therapy Practice Platform</h1>
</div>

## Overview

OPSC helps therapists set up and operate a practice from onboarding to billing and patient care. It combines credentialing automation, telehealth, payments, a lightweight EHR, and realtime messaging in two delivery modes: self-hosted and cloud multi-tenant.

## Product vision

Create an open-source platform inspired by Alma and Headway, offering:

- A self-hosted guided workflow for therapists who want full control.
- A low-cost managed multi-tenant SaaS for therapists who prefer a hosted option.

## Key differentiators

- Credentialing automation using RAG/document extraction to reduce manual application work.
- HIPAA-aware realtime messaging with moderation, audit trails, and optional ML assistance.
- Extensible integrations: Zoom/Google Meet for telehealth, Stripe for payments, Optum for claims.

## 1. Prisma migration (10-step plan)

This project is tracking a planned migration from TypeORM to Prisma. The high-level 10-step plan is maintained in the repo TODOs and reproduced here for visibility so contributors can follow a single source of truth.

1.1 Prep and safety

- Create a branch `feature/prisma-migration` (we use `feature/prisma-migration-1.1` for the initial POC).
- Keep TypeORM in place while migrating incrementally. Back up production DB and test migrations in staging before deploying to production.

1.2 Install Prisma & initialize

- Install `prisma` (dev) and `@prisma/client` (runtime).
- Run `npx prisma init` to create the `prisma/` folder and initial config.

1.3 Create initial Prisma schema

- Prefer `npx prisma db pull` to introspect the existing database and generate `schema.prisma` for the current schema.
- Optionally author `schema.prisma` by hand when restructuring models.

1.4 Generate client

- Run `npx prisma generate` to create the Prisma Client.
- Add a singleton client at `src/lib/prisma.ts` for easy import across services.

1.5 Add Prisma side-by-side

- Use Prisma alongside TypeORM. Implement small Prisma-based services first (start with `User` and `Role`).

1.6 Convert models incrementally (POC)

- Convert `User` and `Role` as a proof-of-concept. Update a small set of tests to use Prisma (SQLite for test runs).

1.7 Data migrations and syncing

- If you modify schema, use `prisma migrate dev` locally to generate migrations. For production deploys run `prisma migrate deploy` or apply SQL migration scripts in a controlled window.

1.8 Replace TypeORM-specific features

- Reimplement TypeORM subscribers, entity listeners, and lifecycle hooks as application-level hooks or Prisma middleware. Rewrite custom repositories as small service functions around the Prisma client.

1.9 Full switch & cleanup

- Once all code uses Prisma, remove TypeORM dependencies, config, and unused entity files. Update CI and deployment manifests to run Prisma migrations and generate the client.

1.10 Tests & verification

- Run the full test suite, add integration tests for migrated paths, and perform a smoke test against staging data. Document the migration runbook and rollback steps.

_This 10-step plan is also present in the project todo list. If you'd like, I can implement the initial POC (install/init Prisma and convert `User`/`Role`) on the `feature/prisma-migration-1.1` branch._

## High-level epics & features

### 1.1. User & Role Management (Backend)

- [ ] Define final DB schema for User, Role, UserRole (partially complete).
- [ ] Create a database seed script to provision the first SuperAdmin user.
- [ ] Build API endpoints (e.g., `/api/admin/users`) for SuperAdmin to manage user roles.

### 1.2. Credentials Auth

- [ ] Implement `authService.signup` for 'Patient' and 'Therapist' types.
- [ ] Implement `authService.login` using email/password and bcrypt.
- [ ] Build password recovery flow:
  - [ ] API route to request a password reset token (generates token, sends email via nodemailer).
  - [ ] API route to confirm and set a new password using the token.

### 1.3. Passkey (WebAuthn) Auth

- [ ] Implement API endpoints for Passkey registration (initiate and verify).
- [ ] Implement API endpoints for Passkey authentication (initiate and verify).
- [ ] Create UI components in user settings for registering and managing passkeys.

### 1.4. OAuth & Provider Linking

- [ ] Finalize NextAuth.js configuration for Google and Facebook providers.
- [ ] Implement backend logic for `linkProvider` and `unlinkProvider` services.
- [ ] Create UI components in user settings to manage linked OAuth accounts.

### 1.5. Profile Management

- [ ] Define DB schema for `TherapistProfile` (e.g., specialty, license, bio) and `PatientProfile` (e.g., basic info).
- [ ] Create secure API endpoints (GET, PUT) for users to manage their own profiles.

## 2. Real-time Messaging (Priority)

### 2.1. Core Messaging MVP (Phase 1)

- [X] Create `feature/messaging/realtime-init` branch.
- [ ] Add entities: `src/db/entities/Thread.ts`, `Message.ts`, `MessageAudit.ts`.
- [ ] Add deterministic moderation helper in `src/lib/moderation.ts` for test-mode.
- [ ] Add Vitest integration test (sqlite) for moderation → blurred + audit.

### 2.2. WebSocket Transport (Phase 2)

- [ ] Configure a socket.io server (or alternative, see tech stack suggestions).
- [ ] Implement socket connection handshake and authentication (using NextAuth.js JWT).
- [ ] Implement `message:send` event from client, triggering moderation and DB save.
- [ ] Implement `message:receive` event pushed from server to all thread participants.

### 2.3. Messaging Features

- [ ] Implement file uploads (API to generate presigned URLs for MinIO).
- [ ] Add DB schema and socket events for emoji reactions.
- [ ] Implement `user:typing` start/stop socket events.
- [ ] Implement presence system (using Redis or socket state) for away/available/offline.

### 2.4. Compliance & Moderation

- [ ] Integrate live OpenAI moderation API (sync on-send).
- [ ] Ensure `MessageAudit` records are created for all messages, edits, deletions, and moderation actions.

## 3. Payments & Claims

### 3.1. Stripe Integration (Phase 3)

- [ ] Integrate Stripe test-mode SDK.
- [ ] Create an API endpoint to create a Stripe `PaymentIntent` for a specific service.
- [ ] Implement client-side UI using Stripe Elements for a secure checkout form.
- [ ] Implement a Stripe webhook handler to listen for `payment_intent.succeeded` and update internal records.
- [ ] Define DB schema for `Invoice` and `Payment` to link transactions to users.

### 3.2. Optum Claims API

- [ ] Implement an API client service for the Optum API.
- [ ] Create API endpoint for "Eligibility Check" via Optum.
- [ ] Create API endpoint and UI for "Claim Submission" via Optum.
- [ ] Define DB schema for `Claim` to track submission status and history.

## 4. Appointments & Telehealth

### 4.1. Core Scheduling

- [ ] Define DB schema for `Appointment`, `Availability` (therapist's schedule), and `SessionNote`.
- [ ] Create UI (e.g., using react-big-calendar) for therapists to set their availability.
- [ ] Create UI for patients to browse availability and book appointments.
- [ ] Build CRUD API endpoints for managing appointments.

### 4.2. Telehealth Integration (Phase 3)

- [ ] Implement Zoom API client (mocked first) to create meetings.
- [ ] Implement Google Meet API client (mocked first).
- [ ] Associate the generated meeting URL with the `Appointment` entity.
- [ ] Implement webhook handlers for meeting callbacks (e.g., session start/end, participants).

### 4.3. Workflow

- [ ] Implement waitlist logic for fully-booked therapists.
- [ ] Implement a reservation flow (e.g., holding a slot for 10 minutes) using Redis.

## 5. Files, EHR & Search

### 5.1. Secure Storage

- [ ] Configure MinIO/S3 bucket policies for HIPAA compliance (private objects, encryption at rest).
- [ ] Implement API endpoints to generate short-lived presigned URLs for secure uploads and downloads.
- [ ] Create UI for managing session notes and patient-uploaded documents.

### 5.2. EHR Schema

- [ ] Define basic EHR schema (e.g., `PatientIntakeForm`, `TreatmentPlan`, `Diagnosis`).
- [ ] Create UI forms for therapists to perform EHR data entry.

### 5.3. Search

- [ ] Implement full-text search on `SessionNote` and `Message` entities (e.g., using Postgres FTS).
- [ ] Build a secure API endpoint that respects patient privacy and permissions.

## 6. Insurance Credentialing Automation

### 6.1. RAG Pipeline (Phase 4)

- [ ] Set up a vector database (e.g., `pgvector` extension for Postgres).
- [ ] Create a document ingestion pipeline to read PDFs of credentialing requirements, create embeddings, and store them.
- [ ] Implement a RAG query service that takes a therapist's profile and finds relevant credentialing requirements.

### 6.2. Application Workflow

- [ ] Create UI for therapists to upload their professional documents (license, diploma, etc.).
- [ ] Implement logic to build "credentialing bundles" based on RAG-extracted requirements.
- [ ] Define DB schema to track application status, follow-up dates, and expirations.

## 7. Admin & Support Console

### 7.1. Ticketing System

- [ ] Define DB schema for `SupportTicket` and `TicketMessage`.
- [ ] Create UI for patients and therapists to submit support tickets.
- [ ] Build the admin UI to view, assign, and manage ticket priority and status.

### 7.2. Admin Tools

- [ ] Implement document upload and extraction tools for the admin console.
- [ ] Create a user lookup and management interface for support staff.

## 8. Observability & Ops

### 8.1. Logging & Monitoring

- [ ] Implement structured logging (e.g., Pino) for all API routes and services.
- [ ] Set up a logging aggregator (e.g., Grafana Loki, Datadog) to parse and view logs.

### 8.2. Dashboards

- [ ] Create dashboards (e.g., in Grafana) pulling from the Postgres database.
- [ ] Add dashboard for signup metrics.
- [ ] Add dashboard for credentialing success/failure rates.
- [ ] Add dashboard for support ticket metrics (e.g., time to resolution).
- [ ] Add dashboard for user retention.

## Data & ML roadmap

- Use counseling conversation datasets and recent research to build sentiment/engagement models.
- Train lightweight models to classify transcripts and chat messages for clinician support and outcome prediction.
- Design ML pipelines with privacy-first and opt-in data policies for cloud tenants.

## Compliance and security

- Architect for HIPAA compliance: audit trails, encryption in transit & at rest (prod), configurable retention, consent for ML.

## Core integrations

- Zoom & Google Meet (telehealth)
- Stripe (payments)
- Optum API (claims/eligibility)
- MinIO / S3 (object storage)
- OpenAI moderation (content filtering) — sync on-send for realtime messages

## Priority roadmap (3-week horizon)

With a 3-week timeline the pragmatic strategy is to focus on a small, testable MVP that demonstrates realtime messaging + credentialing/workflow scaffolding and core infra.

### Phase 0 (setup & governance — 0.5 day)
- Branching policy, contributor docs, CI gating (`CONTRIBUTING.md` added).

### Phase 1 (48 hours) — Immediate priority
- Build `feature/messaging/realtime-init`:
  - Minimal DB entities (Thread, Message, MessageAudit)
  - Deterministic moderation helper (test mode)
  - One integration test (SQLite in-memory) asserting moderation produces blurred messages + an audit entry

### Phase 2 (week 1)
- Basic socket.io handshake + message send flow with moderation in place
- Presigned uploads for MinIO (dev) with tests

### Phase 3 (week 2)
- Appointment + basic Zoom integration (mocked) and Stripe test-mode integration

### Phase 4 (week 3)
- Simple RAG-based credentialing extraction pipeline and admin support UI

## Concrete 48-hour checklist (todo)

- [ ] Create `feature/messaging/realtime-init` branch from `dev`
- [ ] Add entities: `src/db/entities/Thread.ts`, `Message.ts`, `MessageAudit.ts` (explicit types)
- [ ] Add deterministic moderation helper in `src/lib/moderation.ts` for test-mode
- [ ] Add Vitest integration test (sqlite) for moderation -> blurred + audit
- [ ] Add `DEVELOPMENT.md` with run/test commands and quick docker-compose notes

## How we will operate

- Work in small increments on `feature/*` branches and target `dev` for integration.
- Each PR should be small, include tests for the behavior changed, and pass CI before merging.
- Use `NODE_ENV=test` deterministic behaviors for external integrations in tests (OpenAI, Zoom, Stripe).

## Developer quick commands

Install dependencies and run tests:

```bash
npm ci
npm run test
```

Create the messaging branch locally:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/messaging/realtime-init
```

## Assets

- Add the app logo at `public/mess.png` to render the top-of-README image.

## Next choices

1) I can create `feature/messaging/realtime-init` and commit the minimal entities + initial integration test. (Recommended)
2) I can scaffold a GitHub Actions `ci.yml` that runs tests on PRs.
3) I can add `DEVELOPMENT.md` with local dev and docker-compose instructions.

Tell me which to start and I'll proceed in the chosen feature branch.
------------------------
