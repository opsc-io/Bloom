import { MigrationInterface, QueryRunner } from "typeorm";

export class AddPasswordHash001 implements MigrationInterface {
  name = 'AddPasswordHash001';

  public async up(queryRunner: QueryRunner): Promise<void> {
    // Add a nullable password_hash column compatible with Postgres/CockroachDB
    await queryRunner.query(`ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS password_hash VARCHAR;`);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`ALTER TABLE IF EXISTS users DROP COLUMN IF EXISTS password_hash;`);
  }
}

export default AddPasswordHash001;
