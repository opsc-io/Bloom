# Bloom - Therapy Practice Platform: Project Report

## 1. Project Description

### Overview

Bloom is a comprehensive, open-source therapy practice management platform designed to help therapists establish and operate their practices from initial credentialing through patient care and billing. The platform combines automated credentialing workflows, HIPAA-compliant telehealth capabilities, integrated payment processing, electronic health records (EHR), and real-time secure messaging into a unified, self-hosted solution.

Built on modern cloud-native technologies and deployed on Google Kubernetes Engine (GKE), Bloom provides therapists with complete control and data sovereignty while delivering enterprise-grade reliability, security, and scalability.

### Problem Statement and Motivation

Mental health professionals face significant operational challenges when establishing and managing private practices:

- **Complex Credentialing**: Insurance credentialing is a time-consuming, manual process that requires navigating diverse requirements across multiple payers, often taking months to complete.
- **Fragmented Tools**: Therapists typically rely on multiple disconnected platforms for scheduling, billing, documentation, and patient communication, leading to inefficiency and potential HIPAA compliance gaps.
- **High Costs**: Commercial practice management platforms charge substantial monthly fees per provider, with additional charges for features like telehealth and payment processing.
- **Data Privacy Concerns**: Cloud-hosted solutions require trusting third-party vendors with sensitive patient information, creating compliance and ethical concerns.
- **Technical Barriers**: Setting up secure, compliant infrastructure requires specialized technical knowledge that most therapists lack.

Bloom addresses these challenges by providing a fully integrated, self-hosted platform with infrastructure-as-code that can be deployed on any Kubernetes cluster, giving therapists full control over their data while automating complex workflows.

### Target Users

**Primary Users:**
- **Solo and Small Group Therapists**: Mental health professionals establishing private practices who need comprehensive practice management tools with HIPAA compliance
- **Therapy Group Administrators**: Practice managers overseeing multiple therapists and requiring centralized oversight, analytics, and support tools

**Secondary Users:**
- **Patients**: Individuals seeking mental health services who interact with the platform for appointment scheduling, secure messaging, telehealth sessions, and billing
- **Support Staff**: Administrative personnel who handle ticketing, user management, and operational support

### Key Value Propositions

1. **Credentialing Automation**: RAG-based document extraction reduces manual credentialing work from months to days by automatically parsing payer requirements and building application bundles from therapist credentials
2. **Complete Integration**: Unified platform eliminating the need for multiple subscriptions and reducing workflow fragmentation
3. **Data Sovereignty**: Fully self-hosted solution giving therapists complete control over patient data and compliance posture
4. **HIPAA Compliance**: Built-in audit trails, encryption at rest and in transit, consent management, and automated moderation for secure communication
5. **AI-Assisted Care**: Machine learning models provide real-time sentiment analysis and mental health indicator detection to support clinical decision-making
6. **Cost Efficiency**: Open-source platform with predictable infrastructure costs and no per-provider licensing fees
7. **Production-Ready**: Enterprise-grade high availability, auto-scaling, and comprehensive observability out of the box

### Scope and Boundaries

**In Scope:**
- Authentication and authorization with multi-factor authentication (MFA)
- Patient-therapist assignment and relationship management
- Secure real-time messaging with moderation and reactions
- Appointment scheduling with calendar views
- Telehealth integration (Zoom, Google Meet)
- Payment processing (Stripe)
- Insurance claims submission (Optum API)
- File storage and management (Google Cloud Storage)
- Automated credentialing workflow
- ML-powered message analysis for mental health indicators
- Admin dashboard and analytics
- Support ticket system

**Out of Scope:**
- Full EHR functionality (limited to basic session notes and treatment plans)
- Direct integration with all insurance payers (focused on Optum initially)
- Mobile native applications (responsive web-first approach)
- Multi-tenancy (designed for single-organization deployment)
- On-premise deployment without Kubernetes

## 2. Requirements

### Functional Requirements

**FR1: Authentication and Authorization**
- Multi-factor authentication support (email/password, OAuth, TOTP, passkeys)
- Role-based access control (Admin, Therapist, Patient)
- Session management with configurable timeouts
- Audit logging for all authentication events

**FR2: Secure Messaging**
- Real-time bidirectional communication via WebSocket (Socket.io)
- Message persistence in CockroachDB
- Automatic content moderation using OpenAI API
- Emoji reactions and typing indicators
- File attachment support with secure upload URLs
- Message audit trail for compliance

**FR3: Appointment Management**
- CRUD operations for appointments with role-aware access
- Calendar view with week navigation
- Therapist availability scheduling
- Appointment status tracking (scheduled, completed, cancelled, no-show)
- Integration with telehealth meeting generation

**FR4: Telehealth Integration**
- Zoom and Google Meet API integration
- Automatic meeting URL generation and association with appointments
- Webhook handlers for session lifecycle events

**FR5: Credentialing Automation**
- Document upload and extraction pipeline
- RAG-based requirement matching using vector embeddings
- Automated credentialing bundle generation
- Status tracking and expiration management

**FR6: Payment Processing**
- Stripe test-mode and production integration
- Payment intent creation for services
- Secure checkout with Stripe Elements
- Webhook handling for payment confirmations
- Invoice and payment record management

**FR7: Insurance Claims**
- Optum API integration for eligibility verification
- Claims submission and status tracking
- Claim history and audit trail

**FR8: File Management**
- Presigned URL generation for secure uploads to GCS
- File metadata storage and retrieval
- Support for session notes and patient documents

**FR9: Machine Learning Analysis**
- Real-time mental health indicator detection via Vertex AI
- Sentiment analysis, trauma indicators, isolation scores
- Therapist-only visibility with real-time broadcast
- Analysis history and audit trail

**FR10: Administrative Tools**
- User management and lookup
- System metrics dashboard with Grafana integration
- Support ticket management
- Audit log access

### Non-Functional Requirements

**NFR1: HIPAA Compliance**
- Encryption in transit (TLS) and at rest for production environments
- Comprehensive audit trails for all data access and modifications
- Configurable data retention policies
- Explicit patient consent for ML model usage
- Access controls enforcing minimum necessary principle

**NFR2: Scalability**
- Horizontal pod autoscaling (HPA) based on CPU/memory metrics
- Auto-scaling from 2-10 pods in production
- Multi-zone deployment for regional redundancy
- Stateless application design enabling elastic scaling

**NFR3: Availability**
- 99.9% uptime SLO for production environment
- Pod disruption budgets ensuring availability during updates
- CockroachDB 3-node cluster with automatic failover
- Regional GKE Autopilot cluster (us-central1)
- Managed SSL with automatic certificate renewal

**NFR4: Performance**
- p95 response time < 500ms under normal load
- p99 response time < 1000ms
- Health check endpoint p95 < 200ms
- WebSocket message latency p95 < 500ms
- Support for 100+ concurrent users per pod

**NFR5: Security**
- Workload Identity Federation (no service account keys)
- Network policies restricting pod-to-pod communication
- Secret management via GCP Secret Manager
- Regular security scanning with CodeQL
- Content Security Policy (CSP) headers

**NFR6: Observability**
- Structured logging aggregated in Grafana Loki
- Prometheus metrics collection for all components
- Pre-configured Grafana dashboards (overview, CockroachDB, Redis, runtime metrics)
- Distributed tracing for request flows
- Real-time alerting for critical thresholds

**NFR7: Maintainability**
- Infrastructure as code using Kustomize
- Environment-specific overlays (dev, qa, prod)
- Automated CI/CD with GitHub Actions
- Comprehensive documentation
- Type-safe codebase (TypeScript)

### User Stories

**As a Therapist:**
- I want to securely message patients so that I can provide timely support between sessions
- I want to view my appointment calendar so that I can manage my schedule efficiently
- I want to automatically generate credentialing application bundles so that I can reduce time spent on administrative work
- I want to receive real-time mental health indicators from patient messages so that I can identify concerning patterns early
- I want to conduct telehealth sessions via Zoom/Google Meet so that I can provide remote care

**As a Patient:**
- I want to schedule appointments with my therapist so that I can plan my treatment
- I want to securely message my therapist so that I can ask questions between sessions
- I want to join telehealth sessions easily so that I can receive care remotely
- I want to make payments securely so that I can pay for services conveniently

**As an Administrator:**
- I want to view system metrics and dashboards so that I can monitor platform health
- I want to manage support tickets so that I can assist users effectively
- I want to perform user lookups so that I can resolve account issues
- I want to review audit logs so that I can ensure compliance and investigate incidents

### System Constraints

**Technical Constraints:**
- Requires Kubernetes cluster (GKE, EKS, AKS, or self-hosted)
- Minimum cluster resources: 2 CPU cores, 4GB RAM for development
- Requires PostgreSQL-compatible database (CockroachDB or PostgreSQL 12+)
- Requires Redis for caching and pub/sub
- Requires object storage (GCS, S3, or MinIO)

**Operational Constraints:**
- Self-hosted deployment requires technical expertise for setup and maintenance
- Machine learning features require Google Cloud Vertex AI or custom inference infrastructure
- HIPAA compliance requires business associate agreements (BAAs) with cloud providers
- Insurance claims integration limited to Optum API initially

**Regulatory Constraints:**
- Must comply with HIPAA Security and Privacy Rules
- Must maintain audit logs for minimum 6 years
- Must obtain patient consent for ML analysis
- Must implement minimum necessary access controls

## 3. Methodology

### Development Approach

The project follows an **Agile, iterative development methodology** with 3-week sprint cycles focused on delivering incremental value:

**Phase 0: Setup & Governance (0.5 day)**
- Established branching policy (main → qa → dev)
- Created contributor documentation
- Configured CI gating and automated testing

**Phase 1: Messaging MVP (48 hours)**
- Implemented core database entities (Thread, Message, MessageAudit)
- Built deterministic moderation helper for test mode
- Created Vitest integration tests with SQLite in-memory database

**Phase 2: WebSocket & Storage (Week 1)**
- Implemented Socket.io real-time messaging with Redis pub/sub
- Built presigned URL generation for MinIO/GCS
- Added message reactions and typing indicators

**Phase 3: Appointments & Payments (Week 2)**
- Developed appointment CRUD APIs and calendar UI
- Integrated Stripe test-mode for payment processing
- Added Zoom API scaffolding for telehealth

**Phase 4: Credentialing & ML (Week 3)**
- Built RAG-based credentialing extraction pipeline
- Deployed multi-task ML model to Vertex AI
- Created admin support UI and dashboards

**Ongoing Practices:**
- Daily standups for distributed team coordination
- Code reviews required for all pull requests
- Automated CI/CD deployment to dev/qa/prod environments
- Weekly demos to stakeholders
- Retrospectives at end of each sprint

### Technology Selection Rationale

**Frontend: Next.js 16 with App Router**
- Server-side rendering for improved SEO and initial load performance
- API routes eliminate need for separate backend server
- Built-in TypeScript support for type safety
- Large ecosystem and active community

**UI: shadcn/ui (Tailwind + Radix)**
- Accessible, unstyled components meeting WCAG standards
- Composable primitives enabling rapid development
- Full design control without framework lock-in
- Excellent TypeScript support

**Backend: Next.js API Routes + Socket.io**
- Unified codebase reducing context switching
- Simplified deployment (single application)
- WebSocket support via separate Socket.io server for scalability
- Built-in middleware for authentication and error handling

**Database: CockroachDB**
- PostgreSQL compatibility enabling use of Prisma ORM
- Distributed SQL with automatic replication and failover
- Horizontal scalability for future growth
- Strong consistency guarantees for financial transactions
- Self-hostable without vendor lock-in

**Cache: Redis**
- High-performance in-memory caching reducing database load
- Pub/sub for real-time features (typing indicators, presence)
- Session storage for Socket.io
- Simple operational model

**Auth: Better Auth**
- Modern authentication library with Prisma adapter
- Supports multiple providers (email/password, OAuth, passkeys, TOTP)
- Flexible, explicit linking UX preventing accidental account merges
- Strong security defaults suitable for HIPAA environments

**Infrastructure: Google Kubernetes Engine (GKE)**
- Fully managed Kubernetes reducing operational overhead
- Regional Autopilot clusters for high availability
- Workload Identity eliminating need for service account keys
- Integrated load balancing and managed SSL certificates
- Cost-effective for small to medium workloads

**Observability: Prometheus + Grafana + Loki**
- Open-source stack avoiding vendor lock-in
- Prometheus for metrics collection and alerting
- Grafana for visualization with pre-built dashboards
- Loki for log aggregation with LogQL queries
- Promtail DaemonSet for automatic log collection

**ML: Vertex AI + XLM-RoBERTa**
- Managed inference infrastructure with autoscaling
- XLM-RoBERTa Large for multilingual support
- Multi-task learning for efficient training
- Frozen backbone reducing training time and costs

### Architecture Decision Records

**ADR-001: Multi-Process Architecture for WebSocket Scaling**
- **Decision**: Deploy separate Socket.io server process
- **Rationale**: Enables independent scaling of WebSocket connections from HTTP traffic; Redis pub/sub ensures message delivery across pods
- **Consequences**: Requires separate deployment manifest and service; adds Redis as critical dependency

**ADR-002: Client-Side ML Label Recalculation**
- **Decision**: Recalculate mental health labels in application code rather than model output
- **Rationale**: Allows rapid threshold tuning without retraining; model focuses on psychometric scores (more stable)
- **Consequences**: Logic duplication between training and inference; requires synchronization of label determination function

**ADR-003: Workload Identity for GCP Authentication**
- **Decision**: Use Workload Identity instead of service account keys
- **Rationale**: Eliminates secret management overhead; reduces credential exposure; follows GCP security best practices
- **Consequences**: Requires IAM configuration during cluster setup; only works on GKE

**ADR-004: Kustomize for Environment Management**
- **Decision**: Use Kustomize overlays instead of Helm charts
- **Rationale**: Simpler mental model with patches; native kubectl integration; easier to audit configuration changes
- **Consequences**: Less abstraction than Helm; requires more verbose configuration for complex scenarios

**ADR-005: CockroachDB Over PostgreSQL**
- **Decision**: Use CockroachDB as primary database
- **Rationale**: Built-in replication and failover; horizontal scalability; PostgreSQL compatibility; self-hostable
- **Consequences**: Higher resource usage than PostgreSQL; less mature ecosystem; potential edge case incompatibilities

### Deployment Strategy

**Environment Progression:**
```
Development (dev branch) → QA (qa branch) → Production (main branch)
```

**Deployment Pipeline:**
1. Developer pushes to feature branch
2. Automated tests run on GitHub Actions
3. Code review and approval required
4. Merge to dev branch triggers deploy-dev.yml workflow:
   - Builds Docker images (bloom-app, bloom-socket, bloom-db-init)
   - Pushes to Artifact Registry with dev-latest tag
   - Updates GKE deployment using kubectl apply -k k8s/overlays/dev
   - Runs smoke tests
5. QA validation on dev.gcp.bloomhealth.us
6. Merge to qa branch triggers deploy-qa.yml workflow (same process, qa environment)
7. QA approval after manual testing
8. Merge to main branch triggers deploy-prod.yml workflow (production deployment)

**Deployment Verification:**
- Health check endpoint validation
- Smoke tests for critical user flows
- Grafana dashboard monitoring
- Pod rollout status verification

**Rollback Strategy:**
- Use kubectl rollout undo for immediate rollback
- Previous image tags retained in Artifact Registry
- Database migrations must be backward compatible
- Feature flags for gradual rollout of risky features

**Blue-Green Considerations:**
- HPA ensures new pods become ready before old pods terminate
- Pod disruption budgets prevent too many pods from being unavailable
- Readiness probes delay traffic until pod is ready
- Liveness probes restart unhealthy pods

### Testing Strategy

**Unit Testing:**
- Vitest for TypeScript/JavaScript code
- Test-driven development for business logic
- Mocking external dependencies (database, APIs)
- Target: >80% code coverage for critical paths

**Integration Testing:**
- SQLite in-memory database for database-dependent tests
- Testing API endpoints end-to-end
- Redis mock for cache-dependent tests
- Socket.io test client for WebSocket functionality

**Load Testing:**
- k6 scripts in load-tests/ directory
- Health check baseline: 100 VUs, 4 minutes
- API stress test: 0-500 VUs, 8 minutes with spike scenarios
- WebSocket load: 100 concurrent connections, message latency tracking
- Thresholds: p95 < 500ms, error rate < 1%
- Run against QA before production deployments

**Manual Testing:**
- QA checklist for critical user flows
- Exploratory testing in qa environment
- Accessibility testing with screen readers
- Cross-browser compatibility (Chrome, Firefox, Safari, Edge)

**Security Testing:**
- Automated vulnerability scanning with npm audit
- CodeQL analysis in CI pipeline
- Dependency vulnerability checks
- Penetration testing before production launch

**Monitoring in Production:**
- Synthetic monitoring for critical endpoints
- Real user monitoring (RUM) for performance
- Error tracking and aggregation
- Alert thresholds for SLO violations

## 4. Project Goals

### Short-Term Deliverables (3 Months)

**Core Platform:**
- ✅ Complete authentication system with OAuth and MFA
- ✅ Secure real-time messaging with moderation
- ✅ Appointment scheduling with calendar UI
- ✅ File upload to GCS with presigned URLs
- ✅ ML-powered message analysis deployed to Vertex AI
- ✅ Production GKE deployment with HA
- ✅ Observability stack (Prometheus, Grafana, Loki)
- ⏳ Stripe payment integration (in progress)
- ⏳ Basic admin dashboard with metrics
- ⏳ Support ticket system

**Infrastructure:**
- ✅ Dev, QA, and Production environments
- ✅ Automated CI/CD pipeline
- ✅ Infrastructure as code with Kustomize
- ✅ Auto-scaling and pod disruption budgets
- ⏳ Comprehensive monitoring and alerting
- ⏳ Disaster recovery procedures

**Documentation:**
- ✅ Technical documentation (README, k8s/README, ml-inference-flow)
- ✅ Load testing suite with k6
- ✅ Contributor guidelines
- ⏳ User documentation and help center
- ⏳ API documentation (OpenAPI/Swagger)

### Long-Term Vision (6-12 Months)

**Product Evolution:**
- Full EHR functionality with treatment plans and diagnosis tracking
- Credentialing automation with RAG-based requirement extraction
- Optum API integration for insurance claims and eligibility
- Multi-provider Zoom and Google Meet integration with webhooks
- Advanced search across messages and session notes
- Patient portal with appointment booking and document access
- Mobile-responsive design optimizations

**Platform Maturity:**
- Multi-tenancy support for SaaS offering
- White-label customization options
- Marketplace for third-party integrations
- Enhanced ML models for outcome prediction
- Automated backup and restore procedures
- Geographic redundancy across multiple regions

**Community Growth:**
- Open-source community building
- Plugin architecture for extensibility
- Comprehensive developer documentation
- Example deployments for AWS (EKS) and Azure (AKS)
- Helm charts for simplified deployment
- Regular release cadence with semantic versioning

**Compliance & Certifications:**
- HIPAA compliance attestation
- SOC 2 Type II certification
- HITRUST CSF certification
- State-by-state regulatory compliance verification
- Privacy impact assessments

### Success Criteria

**Technical Metrics:**
- 99.9% uptime SLO achieved consistently
- p95 response time < 200ms under normal load
- Zero critical security vulnerabilities
- <1% error rate in production
- Successful disaster recovery drill with <1 hour RTO

**User Adoption:**
- 10 pilot practices using Bloom in production
- 100+ active patient users
- 90% user satisfaction score (NPS > 50)
- <5% monthly churn rate
- Average 20+ messages per patient-therapist relationship monthly

**Business Impact:**
- Credentialing time reduced from 90 days to <30 days
- 50% reduction in administrative time spent on scheduling and billing
- 80% of users report improved practice efficiency
- Successfully process 100+ claims through Optum integration
- Positive ROI for practices compared to commercial alternatives

**Community Indicators:**
- 10+ external contributors to open-source repository
- 100+ GitHub stars
- Active community forum with daily engagement
- 5+ third-party integrations or plugins
- Documentation viewed by 500+ unique visitors monthly

## References

- [README.md](../README.md) - Main project documentation with tech stack and features
- [k8s/README.md](../k8s/README.md) - Kubernetes infrastructure and deployment guide
- [ML Inference Flow](ml-inference-flow.md) - Machine learning architecture and model details
- [Load Testing Documentation](../load-tests/README.md) - Performance testing and SLO targets
- [Contributing Guidelines](../CONTRIBUTING.md) - Development workflow and coding standards
- [Architecture Diagram](../README.md#architecture) - System architecture overview
- [Deployment Workflow](../README.md#cicd-pipeline) - CI/CD pipeline and deployment process

---

*Document Version: 1.0*  
*Last Updated: December 2024*  
*Status: Living Document*
