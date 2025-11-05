import { Entity, PrimaryColumn, Column, OneToOne, ManyToOne, JoinColumn } from 'typeorm';
import { User } from './User';
import { Provider } from './Provider';

@Entity('clients')
export class Client {
  @PrimaryColumn('uuid')
  user_id: string;

  @Column('uuid', { nullable: true })
  primary_therapist: string;

  @Column({ length: 50, nullable: true })
  insurance_id: string;

  @Column({ type: 'jsonb', nullable: true })
  emergency_contact: any;

  @OneToOne(() => User)
  @JoinColumn({ name: 'user_id' })
  user: User;

  @ManyToOne(() => Provider)
  @JoinColumn({ name: 'primary_therapist' })
  provider: Provider;
}
