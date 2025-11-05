// Centralized configuration values. Keep minimal here and read from environment
export const RP_ID = process.env.RP_ID || 'localhost';
export const ORIGIN = process.env.ORIGIN || 'http://localhost:3000';

// Small helper to expose whether we're in development
export const IS_DEV = (process.env.NODE_ENV || 'development') === 'development';
