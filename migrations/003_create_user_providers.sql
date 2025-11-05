-- Migration: create user_providers table to store OAuth provider linkages
CREATE TABLE IF NOT EXISTS user_providers (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  provider VARCHAR(50) NOT NULL,
  provider_user_id VARCHAR(200) NOT NULL,
  provider_data JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- unique constraint to prevent duplicate provider-user links
CREATE UNIQUE INDEX IF NOT EXISTS uniq_user_providers_provider_user ON user_providers (provider, provider_user_id);
