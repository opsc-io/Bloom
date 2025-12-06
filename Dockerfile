FROM node:20-alpine AS base
WORKDIR /app
# Prisma needs OpenSSL + glibc compat on Alpine (Cockroach/Postgres engines)
RUN apk add --no-cache libc6-compat openssl
ENV NEXT_TELEMETRY_DISABLED=1

# Dependencies stage
FROM base AS deps
COPY package*.json ./
# Include Prisma schema/config so postinstall generate succeeds
COPY prisma ./prisma
COPY prisma.config.ts ./
RUN npm ci

# Build stage
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npx prisma generate
RUN npm run build

# Production stage
FROM base AS runner
ENV NODE_ENV=production
# Default URLs can be overridden at runtime
ENV DATABASE_URL="postgresql://root@db:26257/bloom?sslmode=disable"
ENV REDIS_URL="redis://redis:6379"

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/prisma ./prisma
EXPOSE 3000
CMD ["node", "server.js"]
