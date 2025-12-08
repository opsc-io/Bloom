# Development Guide

Quick reference for contributors: infrastructure overview, environment setup, and deployment.

## Infrastructure Overview

### Environments

| Environment | Domain | Database | Branch |
|-------------|--------|----------|--------|
| Production | bloomhealth.us | CockroachDB (meek-wallaby-10799) | main |
| QA | qa.bloomhealth.us | CockroachDB (exotic-cuscus-10800) | qa |

### Tech Stack

- **Framework**: Next.js 16 with App Router
- **Database**: CockroachDB Cloud (PostgreSQL compatible)
- **ORM**: Prisma with CockroachDB provider
- **Auth**: Better Auth (email/password + Google OAuth)
- **Hosting**: Vercel
- **File Storage**: Vercel Blob
- **Email**: SMTP2Go (password reset emails)
- **Observability**: Grafana Cloud

## Prerequisites

- Node 18+
- npm

## Installation

```bash
npm ci
```

## Local Development

```bash
npm run dev
```

The app runs at http://localhost:3000

## Environment Variables

Required variables for local development (create `.env.local`):

```bash
# Database
DATABASE_URL="postgresql://..."

# Auth
BETTER_AUTH_SECRET="..." # min 32 chars
BETTER_AUTH_URL="http://localhost:3000"

# Google OAuth (optional for local)
GOOGLE_CLIENT_ID="..."
GOOGLE_CLIENT_SECRET="..."

# SMTP (for password reset)
SMTP_USER="bloomhealth.us"
SMTP_PASSWORD="..."

# Blob Storage
BLOB_READ_WRITE_TOKEN="..."

# Grafana (admin dashboard)
GRAFANA_TOKEN_QA="..."
GRAFANA_TOKEN_PRODUCTION="..."
```

## Key Files & APIs

### Authentication
- `src/lib/auth.ts` - Server-side Better Auth config
- `src/lib/auth-client.ts` - Client-side auth hooks
- `src/app/api/auth/[...all]/route.ts` - Auth API routes

### File Upload
- `src/app/api/upload/route.ts` - Blob upload/delete/list API
- `src/lib/blob.ts` - Client utilities for file uploads

Usage:
```tsx
import { uploadFile, deleteFile, listFiles } from '@/lib/blob'

const result = await uploadFile(file, 'avatars')
console.log(result.url)
```

### Admin Dashboard
- `src/app/api/admin/stats/route.ts` - User/session statistics
- `src/app/api/admin/grafana/route.ts` - Grafana dashboard URL
- `src/app/dashboard/page.tsx` - Dashboard with admin view

## Database

### Prisma Commands

```bash
# Generate client
npx prisma generate

# Deploy migrations (use on Vercel/CI)
DATABASE_URL="..." npx prisma migrate deploy

# Create migration (local dev)
npx prisma migrate dev --name <migration-name>

# View database
npx prisma studio
```

### Schema Location
- `prisma/schema.prisma` - Main schema file
- `prisma/migrations/` - Migration history

## Deployment

### Production (main branch)
Merges to `main` auto-deploy to bloomhealth.us via Vercel.

### QA (qa branch)
Merges to `qa` auto-deploy to qa.bloomhealth.us via Vercel.

### Manual Deploy
```bash
# QA
VERCEL_TOKEN=... vercel --scope opsc --prod=false
VERCEL_TOKEN=... vercel alias <deployment> qa.bloomhealth.us --scope opsc

# Production
VERCEL_TOKEN=... vercel --scope opsc --prod
VERCEL_TOKEN=... vercel alias <deployment> bloomhealth.us --scope opsc
```

## Branch Workflow

1. Create feature branch from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/<short-desc>
   ```

2. Make changes, commit with prefixes: `feat:`, `fix:`, `chore:`

3. Open PR to `qa` for testing, then to `main` for production

## Vercel Environment Variables

| Variable | Environments | Description |
|----------|--------------|-------------|
| DATABASE_URL | All | CockroachDB connection string |
| BETTER_AUTH_SECRET | All | Auth secret (32+ chars) |
| BETTER_AUTH_URL | Production/Preview | https://bloomhealth.us or preview URL |
| GOOGLE_CLIENT_ID | All | Google OAuth client ID |
| GOOGLE_CLIENT_SECRET | All | Google OAuth client secret |
| BLOB_READ_WRITE_TOKEN | All | Vercel Blob storage token |
| SMTP_USER | All | SMTP2Go username |
| SMTP_PASSWORD | All | SMTP2Go password |
| GRAFANA_TOKEN_QA | Preview | Grafana service account token for QA |
| GRAFANA_TOKEN_PRODUCTION | Production | Grafana service account token for production |

## Observability

Grafana Cloud dashboards are embedded in the admin view at `/dashboard` for users with `administrator=true`.

- Production metrics use `GRAFANA_TOKEN_PRODUCTION`
- QA metrics use `GRAFANA_TOKEN_QA`
