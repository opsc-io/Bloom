import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn, OneToMany } from 'typeorm';
import type { UserRole } from './UserRole';

@Entity('users')
export class User {
  @PrimaryGeneratedColumn('uuid')
  user_id: string;

  @Column({ type: 'citext', unique: true })
  email: string;

  // Store the password hash for traditional email/password auth.
  // Use snake_case column name `password_hash` to match common DB conventions.
  @Column({ name: 'password_hash', nullable: true })
  password_hash: string;

  @Column({ type: 'jsonb', nullable: true })
  passkey_credential: any;

  @Column({ nullable: true })
  mfa_secret: string;

  @CreateDateColumn()
  created_at: Date;

  @Column({ type: 'timestamptz', nullable: true })
  last_login: Date;

  @Column({ default: true })
  is_active: boolean;

  // Use runtime require in the decorator to avoid circular import evaluation.
  @OneToMany(() => require('./UserRole').UserRole, (userRole: UserRole) => userRole.user)
  userRoles: UserRole[];
}
