import { Entity, PrimaryColumn, Column, OneToOne, JoinColumn } from 'typeorm';
import { User } from './User';

@Entity('providers')
export class Provider {
  @PrimaryColumn('uuid')
  user_id: string;

  @Column({ length: 10, unique: true, nullable: true })
  npi_number: string;

  @Column({ 
    type: 'varchar',
    length: 20,
    default: 'pending'
  })
  license_verification_status: string;

  @Column({ type: 'uuid', nullable: true })
  credentialing_package_id: string;

  @OneToOne(() => User)
  @JoinColumn({ name: 'user_id' })
  user: User;
}
