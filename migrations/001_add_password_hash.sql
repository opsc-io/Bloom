-- Migration: Add password_hash column to users table
-- This file is intended to be run against CockroachDB or Postgres-compatible SQL.

ALTER TABLE IF EXISTS users
  ADD COLUMN IF NOT EXISTS password_hash VARCHAR;
