DEVELOPMENT
===========

Quick reference for contributors: how to run tests, start local services, and iterate fast.

Prereqs
- Node 18+ (use nvm to manage versions)
- Docker (optional) for Postgres/Redis/MinIO stack

Install
-------
Install dependencies (use CI-friendly install when scripting):

```bash
npm ci
```

Run tests
---------
Run the entire test suite with Vitest:

```bash
npm run test
```

Run a single test file (fast feedback):

```bash
npx vitest run tests/some.spec.ts
```

Run tests in watch mode while developing:

```bash
npx vitest --watch
```

Testing notes
- The project supports an in-memory SQLite configuration for fast integration tests. Use `NODE_ENV=test` when running tests locally to enable deterministic behaviors for external services (OpenAI moderation, Stripe, Zoom mocks).
- If tests complain about missing drivers (sqlite3), run `npm ci` to ensure dev deps are installed.

Start the app (development)
---------------------------
This repo uses Next.js for the web/server routes. For local dev:

```bash
npm run dev
```

Realtime server
---------------
If you have a standalone realtime server (`src/realtime/server.ts`), start it during development with:

```bash
node -r ts-node/register/transpile-only src/realtime/server.ts
```

or run the compiled build in Node after `npm run build`.

Docker-compose dev stack (optional)
----------------------------------
Use the `docker-compose.yml` in the repo (if present) to start Postgres, Redis and MinIO for realistic dev testing. Example:

```bash
docker compose up -d
# then run migrations or start the app
```

Environment variables
---------------------
Copy `.env.example` to `.env.local` and fill in credentials for Postgres, Redis, MinIO, Stripe test keys, and other integrations. Key variables:

- NODE_ENV=test|development|production
- DATABASE_URL (or DB_HOST/DB_USER/DB_PASS/DB_NAME)
- REDIS_URL
- MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
- STRIPE_SECRET_KEY (for test mode use test key)
- NEXTAUTH_SECRET or JWT_SECRET

Feature branch workflow
-----------------------
1. Create a feature branch from `dev`:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/<short-desc>
```

2. Make small, test-covered commits. Run `npm run test` locally before pushing.
3. Open a PR from `feature/*` -> `dev`. CI will run tests and block merging when failures occur.

Testing external integrations locally
-----------------------------------
- Use deterministic test-modes for external integrations. For example, `src/lib/moderation.ts` should expose a deterministic flag when `NODE_ENV=test` so tests don't call external APIs.
- For payment/Zoom, mock or use provider test modes.

Recovering previously removed files
----------------------------------
If you need to recover uncommitted or deleted local files, you can inspect the git reflog or the shell history. If you want me to attempt to recover previous local edits, tell me and I will try to restore via reflog or reapply patches on a feature branch.

Developer tips
--------------
- Keep commits small and focused. Prefer one behavior change per PR.
- Add tests for new behaviors; prefer unit tests for services and integration tests for DB + services.
- Use `NODE_ENV=test` deterministic behaviors to make CI stable.

That's it — pick a task from the README TODOs and I'll implement it on a feature branch. 
