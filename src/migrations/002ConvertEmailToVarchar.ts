import { MigrationInterface, QueryRunner } from "typeorm";

export class ConvertEmailToVarchar002 implements MigrationInterface {
  name = 'ConvertEmailToVarchar002';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // Create a temporary column and populate with lower(email)
    await queryRunner.query(`ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS email_new VARCHAR;`);
    await queryRunner.query(`UPDATE users SET email_new = lower(CAST(email AS text));`);

    // Drop old column and rename
    await queryRunner.query(`ALTER TABLE users DROP COLUMN IF EXISTS email;`);
    await queryRunner.query(`ALTER TABLE users RENAME COLUMN email_new TO email;`);

    // Create a case-insensitive unique index on lower(email)
    // Use IF NOT EXISTS for idempotency
    await queryRunner.query(`CREATE UNIQUE INDEX IF NOT EXISTS uniq_users_email_lower ON users ((lower(email)));`);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Attempt to restore previous state is non-trivial; as a safe rollback, drop the index
    await queryRunner.query(`DROP INDEX IF EXISTS uniq_users_email_lower;`);
  }
}

export default ConvertEmailToVarchar002;
