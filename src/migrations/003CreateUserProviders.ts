import { MigrationInterface, QueryRunner } from "typeorm";

export class CreateUserProviders003 implements MigrationInterface {
  name = 'CreateUserProviders003';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // Create table user_providers compatible with Postgres/CockroachDB
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS user_providers (
        id UUID PRIMARY KEY,
        user_id UUID NOT NULL,
        provider VARCHAR(50) NOT NULL,
        provider_user_id VARCHAR(200) NOT NULL,
        provider_data JSONB,
        created_at TIMESTAMPTZ DEFAULT now(),
        CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
      );
    `);

    await queryRunner.query(`CREATE UNIQUE INDEX IF NOT EXISTS uniq_user_providers_provider_user ON user_providers (provider, provider_user_id);`);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS uniq_user_providers_provider_user;`);
    await queryRunner.query(`DROP TABLE IF EXISTS user_providers;`);
  }
}

export default CreateUserProviders003;
