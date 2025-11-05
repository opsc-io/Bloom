import { Entity, PrimaryGeneratedColumn, Column, OneToMany } from 'typeorm';
import type { UserRole } from './UserRole';

@Entity('roles')
export class Role {
  @PrimaryGeneratedColumn()
  role_id: number;

  @Column({ length: 50, unique: true })
  name: string;

  // Use runtime require to reference UserRole and avoid circular import issues
  @OneToMany(() => require('./UserRole').UserRole, (userRole: UserRole) => userRole.role)
  userRoles: UserRole[];
}
