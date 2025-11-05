<div align="center">
  <!-- Add `public/mess.png` to the repository to show the app logo here -->
  <img src="/mess.png" alt="App logo" width="160" />
  <h1>OPSC — Therapy Practice Platform</h1>
</div>

## Overview
This repository contains a therapist practice platform backbone: authentication, user management, and (planned) realtime messaging with moderation and file storage.

### Quick status (high level)
- Auth: email/password, password-reset scaffolding, OAuth/provider linking — baseline implemented (see `src/auth`).
- Realtime messaging: scaffolding was prototyped and iterated; current repository baseline does not include the latest uncommitted messaging work — we'll reintroduce it on a feature branch.
- Storage: S3/MinIO presign helpers (planned); email (nodemailer) scaffold added.
- Tests: Unit and integration tests using Vitest run locally with `sqlite` support for quick iteration.

## Where we are headed (3-week target)
- Deliver a minimal, HIPAA-aware realtime messaging MVP: socket.io-based server, synchronous text moderation, message audit logs, and secure file uploads (MinIO for dev, S3 for prod).
- Provide local fast iteration: SQLite for tests, docker-compose for optional Postgres/Redis/MinIO dev environment.

## Tech stack
- Node.js + TypeScript
- Next.js (app routes) for web + API
- TypeORM for DB layer (Postgres in prod, SQLite for tests)
- socket.io for realtime
- MinIO / AWS S3 for object storage (presigned uploads)
- Vitest for tests

## Short-term plan (next 48 hours)
The goal for the next 48 hours is to get a repeatable, safe feature-branch workflow and to reintroduce the messaging scaffold incrementally with passing tests.

### Day 0 (now): Establish branching & repo rules
- Create a `dev` branch and use it as the integration target.
- Add `CONTRIBUTING.md` with branch/PR conventions. (Done)

### Day 1 (next 24 hours): Messaging rehab (iteration 1)
- Create branch `feature/messaging/realtime-init` from `dev`.
- Add minimal messaging entities: `Thread`, `Message`, `MessageAudit` with explicit column types and a small migration or schema sync for local tests.
- Add a deterministic moderation helper (test-mode) and one integration test that runs against an in-memory SQLite DB and asserts the moderation behavior.
- Run tests and iterate until green.

### Day 2 (next 48 hours): Wire realtime and presign
- Add a small `src/realtime/server.ts` that starts socket.io and performs token-based handshake (test tokens allowed in NODE_ENV=test).
- Implement message send flow: save message, run synchronous moderation, set status 'blurred' when flagged, write `MessageAudit` entries.
- Add presign upload endpoint skeleton and storage helper for MinIO (dev) / S3 (prod). Add tests for presign URL generation.

## How we'll work
- Feature branch per unit of work. Small commits, tests run locally before PR.
- CI will run `npm ci` and `npm run test` for every PR (we'll add workflow in next step).
- Keep `main` stable. Use `dev` for merged feature branches, `feature/*` for work in progress.

## Immediate repo TODO (actionable)
1. Create `dev` branch (done by maintainer via git).  
2. Create `feature/messaging/realtime-init` and reapply messaging scaffolding in small, testable commits.  
3. Add CI workflow that runs tests on PRs.  
4. Add `DEVELOPMENT.md` with run/test commands and docker-compose notes.  

## How to run tests locally
Install deps and run vitest:

```bash
npm ci
npm run test
```

## Notes
- The project expects the app logo at `/public/mess.png`. Add that file to the repo `public/` folder to show it in this README. If you want me to add a placeholder image, say the word and I will.

## Contact / planning
I can create the `dev` branch, add CI workflow, and start the `feature/messaging/realtime-init` branch and reapply the messaging work incrementally. Tell me which piece you'd like me to start next (CI, dev branch, or messaging branch) and I'll proceed.
