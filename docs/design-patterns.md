# Design Patterns in Bloom

- **Controller + Repository**: Next.js API routes act as thin controllers delegating to Prisma (single `prisma` client in `src/lib/prisma.ts`) and Redis caches. Example: `src/app/api/messages/route.ts` fetches via `getCachedOrFetch` (cache-aside) and persists through Prisma.
- **Adapter/Facade**: External services are wrapped so callers do not depend on vendor SDKs.
  - `src/lib/ml-inference.ts` hides Vertex AI vs mock inference behind `analyzeText`/`USE_MOCK_ML`.
  - `src/lib/blob.ts` exposes a minimal upload/delete surface over Google Cloud Storage.
  - `src/lib/auth-client.ts` wraps Better Auth client plugins into a typed React auth client.
- **Pub/Sub event-driven**: Redis channels decouple writers from Socket.io delivery. `src/app/api/messages/route.ts` publishes to Redis; `scripts/socket-server.ts` subscribes and fans out events to WebSocket rooms.
- **Cache-aside**: `src/lib/redis.ts::getCachedOrFetch` reads Redis first, falls back to Prisma, then seeds the cache (used in conversations/messages endpoints).
- **Circuit-breaker/fallback lite**: ML inference toggles to deterministic thresholds when `USE_MOCK_ML=true`, preventing upstream ML outages from breaking messaging UX.
- **Stratified routing by feature**: App Router folders (`src/app/messages`, `src/app/admin`, etc.) co-locate UI + server routes, keeping UI/controller/domain boundaries cohesive (micro-frontends not required).
