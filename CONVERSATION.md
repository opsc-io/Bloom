# Bloom Project Setup Conversation

## Overview

This document captures the setup of the Bloom Therapy Practice Platform with Vercel + Kubernetes (CockroachDB) architecture.

---

## 1. Repository Setup

### Cloned from GitHub
```bash
git clone https://github.com/opsc-io/Bloom
```

### Created Branches
```bash
git checkout -b dev
git push -u origin dev

git checkout -b feature/k8s-cockroachdb
git push -u origin feature/k8s-cockroachdb
```

**Branch Structure:**
```
main
  └── dev
       └── feature/k8s-cockroachdb  ← current branch
```

---

## 2. Architecture Decision

### Hybrid Architecture: Vercel + Kubernetes

```
                         PRODUCTION
┌─────────────────────────────────────────────────────────────┐
│                         VERCEL                               │
│              (Next.js + Serverless + CDN)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ DATABASE_URL (TLS)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    KUBERNETES CLUSTER                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           CockroachDB (3-node StatefulSet)           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                         LOCAL DEV
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│  ┌─────────────────┐    ┌─────────────────────────────┐     │
│  │   Next.js Dev   │───▶│  CockroachDB (single node)  │     │
│  │   (npm run dev) │    │       Port 26257            │     │
│  └─────────────────┘    └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Component | Role |
|-----------|------|
| **Vercel** | Hosts Next.js app (auto-scaling, zero-config, CDN) |
| **CockroachDB** | Distributed database (HA, PostgreSQL-compatible) |
| **Kubernetes** | Orchestrates CockroachDB cluster in production |
| **Docker Compose** | Local development with single-node CockroachDB |

---

## 3. Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Next.js 16 (App Router) |
| ORM | Prisma 7 |
| Auth | Better Auth (email/password + Google + Zoom OAuth) |
| Database | CockroachDB (PostgreSQL-compatible) |
| Styling | Tailwind CSS v4 + Shadcn/UI |

---

## 4. CockroachDB vs PostgreSQL

**CockroachDB is NOT PostgreSQL** - it's a different database that uses PostgreSQL's wire protocol for compatibility.

| Feature | PostgreSQL | CockroachDB |
|---------|------------|-------------|
| Architecture | Single-node | Distributed (multi-node) |
| Replication | Leader-follower | Raft consensus |
| Scaling | Vertical | Horizontal |
| High Availability | Requires setup | Built-in |

### Prisma Configuration

Prisma natively supports CockroachDB:
```prisma
datasource db {
  provider = "cockroachdb"  // Native CockroachDB provider
}
```

The connection string uses `postgresql://` protocol because CockroachDB is wire-compatible.

---

## 5. Local Development Setup

### Files Created

#### docker-compose.yml
```yaml
services:
  cockroachdb:
    image: cockroachdb/cockroach:v23.2.0
    container_name: bloom-cockroachdb
    command: start-single-node --insecure --advertise-addr=localhost
    ports:
      - "26257:26257"  # SQL port
      - "8080:8080"    # Admin UI
    volumes:
      - cockroach-data:/cockroach/cockroach-data

  cockroachdb-init:
    image: cockroachdb/cockroach:v23.2.0
    depends_on:
      cockroachdb:
        condition: service_healthy
    command: sql --insecure --host=cockroachdb -e "CREATE DATABASE IF NOT EXISTS bloom;"

volumes:
  cockroach-data:
```

#### .env
```bash
DATABASE_URL="postgresql://root@localhost:26257/bloom?sslmode=disable"
BETTER_AUTH_SECRET="dev-secret-change-in-production"
```

#### prisma/schema.prisma (updated)
```prisma
datasource db {
  provider = "cockroachdb"
}
```

---

## 6. Prisma 7 Configuration

In Prisma 7, the database URL is configured in `prisma.config.ts`, NOT in `schema.prisma`:

### prisma.config.ts
```typescript
import path from "node:path";
import { defineConfig, env } from "prisma/config";
import "dotenv/config";

export default defineConfig({
  schema: path.join("prisma", "schema.prisma"),
  migrations: {
    path: path.join("prisma", "migrations"),
  },
  datasource: {
    url: env("DATABASE_URL"),
  },
});
```

### prisma/schema.prisma
```prisma
generator client {
  provider = "prisma-client"
  output   = "../src/generated/prisma"
  moduleFormat = "esm"
  generatedFileExtension = "ts"
  importFileExtension = "ts"
}

datasource db {
  provider = "cockroachdb"
}
```

---

## 7. Commands Run

### Start CockroachDB
```bash
docker-compose up -d
```

### Run Prisma Migrations
```bash
# Removed old PostgreSQL migrations
rm -rf prisma/migrations

# Created new CockroachDB migrations
npx prisma migrate dev --name init
```

### Generate Prisma Client
```bash
npx prisma generate
```

### Build & Run
```bash
npm run build
npm run dev
```

---

## 8. Running Services

| Service | URL | Status |
|---------|-----|--------|
| Next.js App | http://localhost:3000 | ✅ Running |
| CockroachDB SQL | localhost:26257 | ✅ Running |
| CockroachDB Admin UI | http://localhost:8080 | ✅ Running |

### Available Pages
- `/` - Home
- `/sign-in` - Sign in
- `/sign-up` - Sign up
- `/dashboard` - Dashboard

---

## 9. Project Structure

```
Bloom/
├── prisma/
│   ├── schema.prisma          # CockroachDB provider
│   └── migrations/            # CockroachDB migrations
├── src/
│   ├── app/
│   │   ├── api/auth/          # Better Auth API
│   │   ├── dashboard/         # Dashboard page
│   │   ├── sign-in/           # Sign in page
│   │   ├── sign-up/           # Sign up page
│   │   └── page.tsx           # Home
│   ├── components/ui/         # Shadcn components
│   ├── generated/prisma/      # Generated Prisma client
│   └── lib/
│       ├── auth.ts            # Better Auth config
│       ├── auth-client.ts     # Client auth
│       └── prisma.ts          # Prisma client
├── docker-compose.yml         # Local CockroachDB
├── .env                       # Environment variables
├── prisma.config.ts           # Prisma 7 config
└── package.json               # name: "bloom"
```

---

## 10. Production Configuration

### Domain
**Production URL:** https://bloomhealth.us

### Kubernetes Manifests to Create
```
k8s/
├── namespace.yaml
├── cockroachdb/
│   ├── statefulset.yaml    # 3-node cluster
│   ├── service.yaml        # LoadBalancer
│   └── pvc.yaml
├── secrets.yaml
└── jobs/
    └── migration-job.yaml
```

### Vercel Setup
1. Connect GitHub repo to Vercel
2. Add custom domain: `bloomhealth.us`
3. Set environment variables:
   - `DATABASE_URL` → K8s CockroachDB endpoint
   - `BETTER_AUTH_SECRET` → Generate with `openssl rand -base64 32`
   - `BETTER_AUTH_URL` → `https://bloomhealth.us`
   - `GOOGLE_CLIENT_ID` → From Google Cloud Console
   - `GOOGLE_CLIENT_SECRET` → From Google Cloud Console
   - `ZOOM_CLIENT_ID` → From Zoom Developer Portal
   - `ZOOM_CLIENT_SECRET` → From Zoom Developer Portal

### OAuth Callback URLs
Configure these in your OAuth providers:

| Provider | Callback URL |
|----------|--------------|
| Google | `https://bloomhealth.us/api/auth/callback/google` |
| Zoom | `https://bloomhealth.us/api/auth/callback/zoom` |

### Production Environment Variables
```bash
# Vercel Dashboard → Settings → Environment Variables

DATABASE_URL=postgresql://user:pass@<k8s-cockroachdb-ip>:26257/bloom?sslmode=verify-full
BETTER_AUTH_SECRET=<generate-with-openssl>
BETTER_AUTH_URL=https://bloomhealth.us
GOOGLE_CLIENT_ID=<your-google-client-id>
GOOGLE_CLIENT_SECRET=<your-google-client-secret>
ZOOM_CLIENT_ID=<your-zoom-client-id>
ZOOM_CLIENT_SECRET=<your-zoom-client-secret>
```

---

## 11. Quick Reference

### Local Development
```bash
# Start everything
docker-compose up -d
npm run dev

# Stop CockroachDB
docker-compose down

# Reset database
docker-compose down -v
docker-compose up -d
npx prisma migrate dev
```

### Useful URLs

**Local Development:**
- App: http://localhost:3000
- CockroachDB UI: http://localhost:8080

**Production:**
- App: https://bloomhealth.us
- GitHub: https://github.com/opsc-io/Bloom

---

## 12. Session Update: Merged Varad's Changes

### Changes from main branch (f92bd52)

Varad added new features:
- Dashboard with sidebar navigation
- Sidebar components (`app-sidebar.tsx`, nav components)
- UI components (avatar, breadcrumb, dropdown-menu, sheet, sidebar, skeleton, tooltip)
- Prisma migration for `firstname`/`lastname` fields
- Mobile hook (`use-mobile.ts`)

### Merged to dev branch
```bash
git fetch origin
git checkout dev
git merge origin/main
```

### Pending Action
Database reset needed to sync migrations:
```bash
npx prisma migrate reset --force
npx prisma generate
npm run dev
```

**Note:** This will delete all local dev data (which is fine for development).

### Current Status
- On branch: `dev`
- Local changes stashed and restored
- Database reset completed

---

## 13. CockroachDB Cloud Infrastructure

### Environments

| Environment | Cluster Host | Database | Region |
|-------------|--------------|----------|--------|
| **Production** | `meek-wallaby-10799.jxf.gcp-us-west2.cockroachlabs.cloud` | `bloom` | GCP US West 2 |
| **QA** | `exotic-cuscus-10800.jxf.gcp-us-west2.cockroachlabs.cloud` | `bloom` | GCP US West 2 |
| **Local Dev** | `localhost:26257` | `bloom` | Docker |

### Environment Files (NOT in git - contains secrets)

| File | Environment | Notes |
|------|-------------|-------|
| `.env` | Local Development | Docker CockroachDB |
| `.env.qa` | QA | CockroachDB Cloud |
| `.env.production` | Production | CockroachDB Cloud |

### Connection String Format
```
postgresql://<user>:<password>@<host>:26257/bloom?sslmode=verify-full
```

### SSL Certificates
CockroachDB Cloud uses `sslmode=verify-full` with system CA certificates. No manual cert download required.

### Running Migrations

```bash
# Local (default .env)
npx prisma migrate dev

# QA
DATABASE_URL="<qa-connection-string>" npx prisma migrate deploy

# Production
DATABASE_URL="<prod-connection-string>" npx prisma migrate deploy
```

### Vercel Environment Variables

Configure these in Vercel Dashboard → Settings → Environment Variables:

| Variable | Production | QA/Preview |
|----------|------------|------------|
| `DATABASE_URL` | Production connection string | QA connection string |
| `BETTER_AUTH_SECRET` | `openssl rand -base64 32` | `openssl rand -base64 32` |
| `BETTER_AUTH_URL` | `https://bloomhealth.us` | `https://qa.bloomhealth.us` |

### Tables (Both Environments)
- `user` - User accounts
- `session` - Auth sessions
- `account` - OAuth accounts
- `verification` - Email verification tokens
- `_prisma_migrations` - Migration tracking
