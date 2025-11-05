import { DataSource } from "typeorm";
import { User } from "./entities/User";
import { Role } from "./entities/Role";
import { UserRole } from "./entities/UserRole";
import { Provider } from "./entities/Provider";
import { Client } from "./entities/Client";
import { UserProvider } from "./entities/UserProvider";
import AddPasswordHash001 from "../migrations/001AddPasswordHash";
import ConvertEmailToVarchar002 from "../migrations/002ConvertEmailToVarchar";
import CreateUserProviders003 from "../migrations/003CreateUserProviders";
import { IS_DEV } from "../config";

const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_PORT = parseInt(process.env.DB_PORT || '5432', 10);
const DB_USER = process.env.DB_USER || 'postgres';
const DB_PASS = process.env.DB_PASS || 'postgres';
const DB_NAME = process.env.DB_NAME || 'therapy_platform';

export const AppDataSource = new DataSource({
    type: "postgres",
    host: DB_HOST,
    port: DB_PORT,
    username: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    // Use synchronize only in development. Use migrations for production.
    synchronize: IS_DEV,
    logging: IS_DEV,
    entities: [User, Role, UserRole, Provider, Client, UserProvider],
    subscribers: [],
    migrations: [AddPasswordHash001, ConvertEmailToVarchar002, CreateUserProviders003],
});
