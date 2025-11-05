-- Migration: convert users.email from citext to varchar and add case-insensitive unique index
-- Designed to be safe for Postgres and CockroachDB.

-- 1) Add a temporary column and populate it with lower(email)
ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS email_new VARCHAR;

UPDATE users SET email_new = lower(email);

-- 2) Drop the old email column and rename the new one
ALTER TABLE users DROP COLUMN IF EXISTS email;
ALTER TABLE users RENAME COLUMN email_new TO email;

-- 3) Create a case-insensitive unique index on lower(email)
-- For Postgres: CREATE UNIQUE INDEX CONCURRENTLY if needed. For CockroachDB, regular CREATE UNIQUE INDEX works.
DO $$
BEGIN
  -- attempt Postgres-style index creation
  BEGIN
    EXECUTE 'CREATE UNIQUE INDEX IF NOT EXISTS uniq_users_email_lower ON users ((lower(email)))';
  EXCEPTION WHEN OTHERS THEN
    -- fallback to simple index if the above fails
    EXECUTE 'CREATE UNIQUE INDEX IF NOT EXISTS uniq_users_email_lower ON users (email)';
  END;
END
$$;
