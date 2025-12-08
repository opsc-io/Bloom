import { PrismaClient } from "@/generated/prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";

const globalForPrisma = global as unknown as {
  prisma: PrismaClient | undefined;
};

let prisma: PrismaClient;

if (!globalForPrisma.prisma) {
  const adapter = new PrismaPg({
    connectionString: process.env.DATABASE_URL!,
  });
  
  globalForPrisma.prisma = new PrismaClient({
    adapter,
  });
}

prisma = globalForPrisma.prisma;

export default prisma;