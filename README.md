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

Key differentiators
- Credentialing automation using RAG/document extraction to reduce manual application work.
- HIPAA-aware realtime messaging with moderation, audit trails, and optional ML assistance.
- Extensible integrations: Zoom/Google Meet for telehealth, Stripe for payments, Optum for claims.

High-level epics & features
---------------------------
1) Authentication & Profiles
  - Sign-up/sign-in for Patient/Therapist/Support/SuperAdmin
  - Passkeys (WebAuthn), password recovery, provider linking/unlinking

2) Real-time Messaging (Priority)
  - One-to-one and therapist↔support channels
  - File uploads, emoji reactions, typing indicators, presence (away/available/offline)
  - OpenAI moderation (sync on-send) and MessageAudit records for compliance

3) Payments & Claims
  - Stripe integration for payments and billing
  - Claim submission and eligibility via Optum API

4) Appointments & Telehealth
  - Calendar picker, session notes, Zoom/Google Meet integrations (meeting creation + callbacks)
  - Waitlist and reservation flows

5) Files, EHR & Search
  - Secure storage for session notes and documents, full-text search, basic EHR schema

6) Insurance Credentialing Automation
  - RAG-based extraction of credentialing requirements
  - Build and submit credentialing bundles; track application state and expirations

7) Admin & Support Console
  - Ticketing, priority/status management, document upload + extraction

8) Observability & Ops
  - Dashboards: signups, credentialing success/failures, support metrics, retention

Data & ML roadmap
-----------------
- Use counseling conversation datasets and recent research to build sentiment/engagement models.
- Train lightweight models to classify transcripts and chat messages for clinician support and outcome prediction.
- Design ML pipelines with privacy-first and opt-in data policies for cloud tenants.

Compliance and security
-----------------------
- Architect for HIPAA compliance: audit trails, encryption in transit & at rest (prod), configurable retention, consent for ML.

Core integrations
-----------------
- Zoom & Google Meet (telehealth)
- Stripe (payments)
- Optum API (claims/eligibility)
- MinIO / S3 (object storage)
- OpenAI moderation (content filtering) — sync on-send for realtime messages

Priority roadmap (3-week horizon)
---------------------------------
With a 3-week timeline the pragmatic strategy is to focus on a small, testable MVP that demonstrates realtime messaging + credentialing/workflow scaffolding and core infra.

Phase 0 (setup & governance — 0.5 day)
- Branching policy, contributor docs, CI gating (CONTRIBUTING.md added).

Phase 1 (48 hours) — Immediate priority
- Build `feature/messaging/realtime-init`:
  - Minimal DB entities (Thread, Message, MessageAudit)
  - Deterministic moderation helper (test mode)
  - One integration test (SQLite in-memory) asserting moderation produces blurred messages + an audit entry

Phase 2 (week 1)
- Basic socket.io handshake + message send flow with moderation in place
- Presigned uploads for MinIO (dev) with tests

Phase 3 (week 2)
- Appointment + basic Zoom integration (mocked) and Stripe test-mode integration

Phase 4 (week 3)
- Simple RAG-based credentialing extraction pipeline and admin support UI

Concrete 48-hour checklist (todo)
----------------------------------
- [ ] Create `feature/messaging/realtime-init` branch from `dev`
- [ ] Add entities: `src/db/entities/Thread.ts`, `Message.ts`, `MessageAudit.ts` (explicit types)
- [ ] Add deterministic moderation helper in `src/lib/moderation.ts` for test-mode
- [ ] Add Vitest integration test (sqlite) for moderation -> blurred + audit
- [ ] Add `DEVELOPMENT.md` with run/test commands and quick docker-compose notes

How we will operate
--------------------
- Work in small increments on `feature/*` branches and target `dev` for integration.
- Each PR should be small, include tests for the behavior changed, and pass CI before merging.
- Use `NODE_ENV=test` deterministic behaviors for external integrations in tests (OpenAI, Zoom, Stripe).

Developer quick commands
------------------------
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

Assets
------
- Add the app logo at `public/mess.png` to render the top-of-README image.

Next choices
------------
1) I can create `feature/messaging/realtime-init` and commit the minimal entities + initial integration test. (Recommended)
2) I can scaffold a GitHub Actions `ci.yml` that runs tests on PRs.
3) I can add `DEVELOPMENT.md` with local dev and docker-compose instructions.

Tell me which to start and I'll proceed in the chosen feature branch.
