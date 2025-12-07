<div align="center">
  <img src="/logo.svg" alt="App logo" width="160" />
  <h1>Bloom — Therapy Practice Platform</h1>
</div>

## Tech Stack

- **Frontend**: Next.js 16 (App Router, TypeScript), shadcn/ui (Tailwind + Radix)
- **Backend**: Next.js API Routes, Socket.io (real-time messaging)
- **Database**: CockroachDB (self-hosted in Kubernetes)
- **Cache**: Redis (self-hosted StatefulSet)
- **Auth**: Better Auth (Prisma adapter, OAuth, 2FA)
- **Storage**: Google Cloud Storage (file uploads)
- **Observability**: Prometheus, Grafana, Loki, Promtail

## Architecture

Bloom is a fully self-hosted platform running on Google Kubernetes Engine (GKE) with HA, auto-scaling, and a complete observability stack.


<img width="3708" height="1492" alt="image" src="https://github.com/user-attachments/assets/508509c4-053e-4f02-9543-c9c952fa5244" />




### Production HA Features

- **Regional GKE Autopilot**: Multi-zone redundancy in us-central1
- **Auto-scaling**: HPA scales pods 2-10 based on CPU/memory
- **Pod Disruption Budgets**: Ensures availability during updates
- **CockroachDB HA**: 3-node cluster with automatic failover
- **Managed SSL**: Auto-renewed certificates via GKE
- **Workload Identity**: No service account keys needed

### Environments

#### Development
| Resource | Value |
|----------|-------|
| Cluster | `bloom-dev-cluster` (us-central1-a, zonal) |
| URL | https://dev.gcp.bloomhealth.us |
| Branch | `dev` |
| Static IP | `bloom-dev-ip` (34.117.169.137) |
| GCS Bucket | `bloom-uploads-dev` |
| CockroachDB | 1 replica (single node) |
| Workflow | `.github/workflows/deploy-dev.yml` |

#### QA
| Resource | Value |
|----------|-------|
| Cluster | `bloom-qa-cluster` (us-central1, regional Autopilot) |
| URL | https://qa.gcp.bloomhealth.us |
| Branch | `qa` |
| Static IP | `bloom-qa-ip` (34.117.179.36) |
| GCS Bucket | `bloom-uploads-qa` |
| CockroachDB | 1 replica |
| Workflow | `.github/workflows/deploy-qa.yml` |

#### Production
| Resource | Value |
|----------|-------|
| Cluster | `bloom-prod-cluster` (us-central1, regional Autopilot HA) |
| URL | https://bloomhealth.us |
| Branch | `main` |
| Static IP | `bloom-prod-ip` (130.211.29.63) |
| GCS Bucket | `bloom-uploads-prod` |
| CockroachDB | 3 replicas (HA) |
| Workflow | `.github/workflows/deploy-prod.yml` |

### CI/CD

- **GitHub Actions** for automated deployments
- **Workload Identity Federation** for secure GCP authentication (no service account keys)
- **Kustomize** for environment-specific configurations

## Overview

Bloom helps therapists set up and operate a practice from onboarding to billing and patient care. It combines credentialing automation, telehealth, payments, a lightweight EHR, and realtime messaging.

## Product vision

Create an open-source platform inspired by Alma and Headway, offering:

- A fully self-hosted platform for therapists who want full control and data sovereignty.
- Complete infrastructure-as-code for reproducible deployments on any Kubernetes cluster.

## Key differentiators

- Credentialing automation using RAG/document extraction to reduce manual application work.
- HIPAA-aware realtime messaging with moderation, audit trails, and optional ML assistance.
- Extensible integrations: Zoom/Google Meet for telehealth, Stripe for payments, Optum for claims.
.
## Current features

- Email/password auth with Better Auth + Prisma; OAuth scaffolding (Google/Zoom); role selection; profile/password settings; optional 2FA
- Test-mode hash override and bcrypt-based hashing for seeded users
- Dashboard with people, messaging, and calendar cards
- Messaging API with Redis caching, reactions, and socket publish; messages page UI
- Appointment CRUD APIs with role-aware access; calendar UI (week offsets)
- Therapist↔patient assignment model and API (current/past therapist relationships)
- Admin stats and Grafana proxy endpoints with admin dashboard UI
- File upload API (GCS/MinIO)
- People discovery endpoints and pages (available therapists, connections)
- Seed scripts for admin + Faker users, appointments, conversations, and assignments

## High-level epics & features

## 1. Setup: Next.js + Prisma + Better Auth + shadcn/ui

This project will be re-bootstrapped on a minimal, secure foundation: Next.js (App Router + TypeScript), Prisma for the ORM, and Better Auth for authentication. UI primitives come from shadcn/ui (Tailwind + Radix primitives). The sections below capture the immediate setup and authentication design choices we will use for the refactor.

### 1.1 Project bootstrap (commands and artifacts)

Overall goal: Bootstrap the Next.js + Prisma + Better Auth project with minimal artifacts.

- [x] Create a new Next.js app (App Router + TypeScript): `npx create-next-app@latest . --use-npm --ts --app`
- [x] Install Prisma, Better Auth and UI tooling: `npm install -D prisma; npm install @prisma/client better-auth tailwindcss postcss autoprefixer; npx tailwindcss init -p; # add shadcn/ui pieces per the shadcn docs`
- [x] Initialize Prisma and generate the client: `npx prisma init; # set DATABASE_URL in .env to your dev Postgres (recommended); npx prisma db pull; # optional: introspect existing DB; npx prisma generate`
- [x] Create a global Prisma client at `src/lib/prisma.ts` and an auth wiring file at `src/lib/auth.ts` using Better Auth's Prisma adapter. See 1.3 and 1.4 for details.

These steps produce the minimal artifacts we need: `prisma/schema.prisma`, `src/lib/prisma.ts`, `src/lib/auth.ts`, and the Next.js app skeleton.

### 1.2 User & Credentials (high-level contract)

Overall goal: Establish user management with secure authentication flows (email/password, OAuth, passkeys, TOTP) and role-based access.

- [x] Define purpose and success criteria: inputs (email+password, passkey assertion, OAuth token), outputs (valid session + user object), success (full auth flows with audit records).
- [x] Implement design notes: use Better Auth's email+password provider and Prisma adapter for credential storage and sessions; maintain application domain models for Role/UserRole; prefer explicit linking UX for providers to avoid accidental merges.

### 1.3 Better Auth: OAuth helper and session wiring

Overall goal: Wire Better Auth for OAuth providers and session management, persisting to Prisma.

- [x] Use the Better Auth Prisma adapter to persist User, Session, Account, and Verification models; add models to `schema.prisma` and run migrations against a disposable database first.
- [x] Create `src/lib/auth.ts` and export a configured `auth` instance with emailAndPassword enabled and trusted origins.
- [x] Expose the Next.js API route at `src/app/api/auth/[...all]/route.ts` using `toNextJsHandler(auth)` from `better-auth/next-js`.
- [x] Configure OAuth provider credentials via env vars and implement explicit UI flow for linking/unlinking providers with audit entries.
- [x] Ensure security: never enable automatic provider-to-existing-account linking without verification for HIPAA-sensitive workflows.

### 1.4 Passkeys & TOTP (Better Auth plugins)

Overall goal: Enable passkeys (WebAuthn) and TOTP for strong, auditable multi-factor authentication.

- [ ] Enable passkeys: use Better Auth's passkey plugin (or @simplewebauthn/server adapter); persist credential metadata in DB for audit/forensics.
- [ ] Enable TOTP: use Better Auth TOTP plugin with encrypted secrets, recovery codes, and enrollment UI; require 2FA for sensitive operations.
- [ ] Wire plugin options in `src/lib/auth.ts` (e.g., `passkeys: { enabled: true }`, `totp: { enabled: true }`); add extra Prisma models if needed and run migrations in test DB first.
- [ ] Ensure audit records for passkey/TOTP flows (actor, timestamp, device metadata, IP).


## 2. Real-time Messaging (Priority)

### 2.1. Core Messaging MVP (Phase 1)

- [x] Implement conversation/message endpoints backed by Prisma
- [x] Add deterministic user-facing mapping for conversations and last messages
- [ ] Add deterministic moderation helper in `src/lib/moderation.ts` for test-mode.
- [ ] Add Vitest integration test (sqlite) for moderation → blurred + audit.

### 2.2. WebSocket Transport (Phase 2)

- [x] Configure a socket.io server and publish new messages/reactions
- [ ] Implement socket connection handshake and authentication (using NextAuth.js JWT).
- [ ] Implement `message:send` event from client, triggering moderation and DB save.
- [ ] Implement `message:receive` event pushed from server to all thread participants.

### 2.3. Messaging Features

- [ ] Implement file uploads (API to generate presigned URLs for MinIO).
- [x] Add DB schema and socket events for emoji reactions.
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

- [x] Define DB schema for `Appointment`
- [ ] Define DB schema for `Availability` (therapist's schedule) and `SessionNote`.
- [x] Create UI for patients/therapists to view appointments (calendar/dashboard)
- [ ] Create UI for therapists to set their availability.
- [ ] Create UI for patients to browse availability and book appointments.
- [x] Build CRUD API endpoints for managing appointments.

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
- [x] Implement API endpoints to generate short-lived upload URLs (GCS/MinIO) and store metadata.
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

- [x] Define DB schema for `SupportTicket` and related ticket attachments
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

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

## Deployment

### CI/CD Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │     │   GitHub    │     │  Artifact   │     │     GKE     │
│    Push     │────▶│   Actions   │────▶│  Registry   │────▶│   Cluster   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      │              Build Docker        Push Images          kubectl apply
      │              Run Tests           bloom-app             Kustomize
      │                                  bloom-socket          overlay
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Branch → Environment                            │
├─────────────────────────────────────────────────────────────────────────┤
│  dev branch   ──▶  bloom-dev-cluster   ──▶  dev.gcp.bloomhealth.us     │
│  qa branch    ──▶  bloom-qa-cluster    ──▶  qa.gcp.bloomhealth.us      │
│  main branch  ──▶  bloom-prod-cluster  ──▶  bloomhealth.us             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow Files

| Workflow | Branch | Cluster | Domain |
|----------|--------|---------|--------|
| `.github/workflows/deploy-dev.yml` | `dev` | bloom-dev-cluster | dev.gcp.bloomhealth.us |
| `.github/workflows/deploy-qa.yml` | `qa` | bloom-qa-cluster | qa.gcp.bloomhealth.us |
| `.github/workflows/deploy-prod.yml` | `main` | bloom-prod-cluster | bloomhealth.us |

### Prerequisites

- GCP Project with GKE API enabled
- Docker for building images
- kubectl and gcloud CLI installed
- Workload Identity Federation configured

### Local Development

```bash
# Start local services (CockroachDB, Redis)
docker-compose up -d

# Run the app
npm run dev
```

### Deploy to GKE (Automated)

Simply push to the appropriate branch:

```bash
# Deploy to dev
git checkout dev && git push origin dev

# Deploy to QA (via PR merge)
git checkout qa && git merge dev && git push origin qa

# Deploy to production (via PR merge)
git checkout main && git merge qa && git push origin main
```

### GCS Storage Buckets

| Environment | Bucket Name |
|-------------|-------------|
| Dev | `bloom-uploads-dev` |
| QA | `bloom-uploads-qa` |
| Prod | `bloom-uploads-prod` |

**Environment Variables:**
```bash
GCS_BUCKET_NAME=bloom-uploads-dev  # or qa/prod per environment
# No credentials needed in GKE - uses Workload Identity
```

**For local development:**
```bash
# Create service account key
gcloud iam service-accounts keys create gcs-key.json \
  --iam-account=github-actions-deploy@project-4fc52960-1177-49ec-a6f.iam.gserviceaccount.com

# Set env var
export GOOGLE_APPLICATION_CREDENTIALS=./gcs-key.json
```

### Manual Deployment

```bash
# Build and push images
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/bloom-app:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/bloom-images/bloom-app:latest

# Deploy to cluster
kubectl apply -k k8s/overlays/dev
```

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [CockroachDB Docs](https://www.cockroachlabs.com/docs/)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
