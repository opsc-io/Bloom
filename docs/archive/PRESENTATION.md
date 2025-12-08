# Bloom Health - Project Presentation

## 1. Quick Summary

**Bloom Health** is a HIPAA-ready therapy practice management platform that connects therapists with patients for mental health services.

### Core Features
- **Patient Portal**: Schedule appointments, message therapists, join video sessions
- **Therapist Dashboard**: Manage patients, view schedules, conduct sessions
- **Admin Dashboard**: User analytics, growth metrics, system monitoring
- **Secure Authentication**: Email/password, Google SSO, Zoom SSO, 2FA/TOTP

### Target Users
- Mental health therapists and counselors
- Patients seeking therapy services
- Practice administrators

---

## 2. Tech Stack & Architecture

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16 | React framework with App Router |
| React | 19 | UI library with Server Components |
| TailwindCSS | 4 | Utility-first styling |
| shadcn/ui | Latest | Component library |
| Recharts | 3.5 | Data visualization |
| Lucide | Latest | Icons |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js API Routes | 16 | RESTful API endpoints |
| Prisma ORM | 7 | Type-safe database access |
| Better Auth | 1.4 | Authentication & sessions |
| Nodemailer | 7 | Email delivery |

### Infrastructure
| Service | Provider | Purpose |
|---------|----------|---------|
| Hosting | Vercel | Edge deployment, serverless functions |
| Database | CockroachDB Cloud | Distributed SQL (2 clusters: QA + Prod) |
| Observability | Grafana Cloud | Metrics dashboards |
| Video | Zoom SDK | Video conferencing |
| Container | Docker | Local development, portable deployment |

### Architecture Highlights
- **Serverless**: No servers to manage, auto-scaling
- **Edge Network**: Global CDN, DDoS protection, SSL/TLS
- **Multi-tenant DB**: Separate QA and Production databases
- **Type Safety**: End-to-end TypeScript with Prisma

```
[Browser] → [Vercel Edge] → [Next.js API] → [Prisma] → [CockroachDB]
                ↓
         [Better Auth]
                ↓
    [Google OAuth | Zoom OAuth | Email/2FA]
```

---

## 3. Learnings

### What Went Great ✅

1. **Prisma + CockroachDB Integration**
   - Type-safe queries with excellent DX
   - CockroachDB's PostgreSQL compatibility made migration seamless
   - Separate QA/Prod clusters provide safe testing

2. **Better Auth**
   - Unified auth for email, OAuth, and 2FA
   - Session management just works
   - Easy to extend with custom fields (therapist, administrator)

3. **Vercel Deployment**
   - Zero-config deployments
   - Preview URLs for every PR
   - Environment variable management per branch

4. **shadcn/ui + TailwindCSS 4**
   - Rapid UI development
   - Consistent design system
   - Great accessibility defaults

5. **Recharts for Admin Dashboard**
   - Embedded charts instead of iframe dependencies
   - Full control over styling and data

### Challenges Faced ⚠️

1. **2FA Implementation**
   - TOTP setup required careful state management
   - QR code generation had edge cases with URL encoding
   - Recovery codes workflow needed multiple iterations

2. **Environment Configuration**
   - Managing env vars across local/QA/prod was complex
   - SMTP credentials initially not loading correctly
   - Solved with proper Vercel env var setup

3. **CockroachDB Migrations**
   - Some PostgreSQL-specific features not supported
   - Had to adjust indexes and constraints
   - `prisma db push` vs `migrate deploy` decisions

4. **TypeScript Strictness**
   - `percent` possibly undefined in Recharts
   - Session type casting for custom fields
   - Prisma generated types occasionally stale

### What I Would Do Differently 🔄

1. **Start with a Monorepo**
   - Separate packages for shared types, UI components
   - Better code organization as project grows

2. **Add End-to-End Tests Earlier**
   - Playwright or Cypress from day one
   - Critical paths: auth flow, appointment booking

3. **Implement Feature Flags**
   - Gradual rollout of features
   - Easier A/B testing
   - Quick rollback capability

4. **Use tRPC or GraphQL**
   - Type-safe API calls end-to-end
   - Better client-side caching
   - Auto-generated API documentation

5. **Redis/Caching from Start**
   - Session storage in Redis
   - API response caching
   - Rate limiting infrastructure

---

## 4. Live Demo Script

### Demo Flow (5-7 minutes)

**1. Landing Page** (30s)
- Show responsive design
- Highlight CTA buttons

**2. Authentication** (1m)
- Sign up with email
- Show email verification
- Demonstrate Google OAuth
- Enable 2FA with authenticator app

**3. Patient Experience** (2m)
- Complete onboarding questionnaire
- View dashboard
- Browse available therapists
- Schedule an appointment
- Send a message

**4. Therapist Experience** (1.5m)
- Login as therapist
- View patient list
- Check appointment calendar
- Reply to messages
- Start Zoom session

**5. Admin Dashboard** (1m)
- Login as admin
- Show user growth chart
- View user distribution pie chart
- Recent signups table
- Grafana integration link

### Demo URLs
- Production: https://bloomhealth.us
- QA: https://qa.bloomhealth.us
- Local: http://localhost:3000

### Demo Accounts
```
Admin: admin@bloomhealth.us / Admin1234!
```

---

## 5. Project Artifacts

### Repository
- **GitHub**: github.com/opsc-io/bloom (private)
- **Branches**: main (prod), qa (staging), dev (development)

### Documentation
| File | Description |
|------|-------------|
| [README.md](README.md) | Project overview and setup |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture diagram |
| [openapi.yaml](openapi.yaml) | API specification |
| [PRESENTATION.md](PRESENTATION.md) | This document |

### Deployed Environments
| Environment | URL | Database |
|-------------|-----|----------|
| Production | bloomhealth.us | meek-wallaby (CockroachDB) |
| QA/Staging | qa.bloomhealth.us | exotic-cuscus (CockroachDB) |

### Key Directories
```
bloom/
├── src/
│   ├── app/              # Next.js App Router pages
│   │   ├── api/          # REST API endpoints
│   │   ├── admin/        # Admin dashboard
│   │   ├── dashboard/    # User dashboard
│   │   └── settings/     # User settings
│   ├── components/       # React components
│   └── lib/              # Utilities (auth, prisma, etc.)
├── prisma/
│   ├── schema.prisma     # Database schema
│   └── seed-admin.ts     # Admin user seeder
├── docker-compose.yml    # Local development
├── Dockerfile            # Production container
└── openapi.yaml          # API documentation
```

### Infrastructure Status
| Component | Status |
|-----------|--------|
| REST APIs | ✅ Complete |
| SSO (Google, Zoom) | ✅ Complete |
| 2FA/TOTP | ✅ Complete |
| Git Repository | ✅ Complete |
| CockroachDB | ✅ Complete |
| BI Dashboard (Grafana) | ✅ Complete |
| Docker Deployment | ✅ Complete |
| Admin Dashboard | ✅ Complete |
| Caching (Redis) | ✅ Complete |
| Observability (Grafana) | ✅ Complete |
| CI/CD (Vercel) | ✅ Complete |
| Queues/Background Jobs | ❌ Future |

---

## Contact

- **Project Lead**: [Your Name]
- **GitHub**: github.com/opsc-io/bloom
- **Production**: https://bloomhealth.us
