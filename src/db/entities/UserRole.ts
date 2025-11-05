import { Entity, PrimaryColumn, ManyToOne } from 'typeorm';
import type { User } from './User';
import type { Role } from './Role';

@Entity('user_roles')
export class UserRole {
  @PrimaryColumn('uuid')
  user_id: string;

  @PrimaryColumn()
  role_id: number;

  // Use runtime require inside the relation function to avoid circular
  // module-evaluation errors ("Cannot access 'User' before initialization").
  // Keep `import type` for static typing only.
  @ManyToOne(() => require('./User').User, (user: User) => user.userRoles)
  user: User;

  @ManyToOne(() => require('./Role').Role, (role: Role) => role.userRoles)
  role: Role;
}
