import { createSwaggerSpec } from "next-swagger-doc";

export const getApiDocs = async () => {
  const spec = createSwaggerSpec({
    apiFolder: "src/app/api",
    definition: {
      openapi: "3.0.0",
      info: {
        title: "Bloom API",
        version: "1.0.0",
        description:
          "API documentation for Bloom - Therapy Practice Platform. All endpoints require authentication via session cookie unless otherwise noted.",
        contact: {
          name: "Bloom Support",
          url: "https://bloomhealth.us",
        },
      },
      servers: [
        {
          url: "https://bloomhealth.us",
          description: "Production",
        },
        {
          url: "https://qa.gcp.bloomhealth.us",
          description: "QA",
        },
        {
          url: "https://dev.gcp.bloomhealth.us",
          description: "Development",
        },
        {
          url: "http://localhost:3000",
          description: "Local Development",
        },
      ],
      tags: [
        { name: "Auth", description: "Authentication endpoints (Better Auth)" },
        { name: "User", description: "User profile and settings management" },
        { name: "Appointments", description: "Appointment scheduling and management" },
        { name: "Messages", description: "Real-time messaging and conversations" },
        { name: "Therapists", description: "Therapist discovery and connections" },
        { name: "Admin", description: "Administrator dashboard and statistics" },
        { name: "Files", description: "File upload and storage" },
        { name: "Health", description: "Health check endpoints" },
      ],
      components: {
        securitySchemes: {
          sessionAuth: {
            type: "apiKey",
            in: "cookie",
            name: "better-auth.session_token",
            description: "Session cookie set by Better Auth after login",
          },
        },
        schemas: {
          Error: {
            type: "object",
            properties: {
              error: { type: "string", description: "Error message" },
            },
            required: ["error"],
          },
          User: {
            type: "object",
            properties: {
              id: { type: "string", format: "uuid" },
              email: { type: "string", format: "email" },
              firstname: { type: "string", nullable: true },
              lastname: { type: "string", nullable: true },
              role: { type: "string", enum: ["PATIENT", "THERAPIST", "UNSET"] },
              therapist: { type: "boolean" },
              administrator: { type: "boolean" },
              createdAt: { type: "string", format: "date-time" },
            },
          },
          Appointment: {
            type: "object",
            properties: {
              id: { type: "string", format: "uuid" },
              title: { type: "string" },
              start: { type: "string", format: "date-time" },
              end: { type: "string", format: "date-time" },
              durationMinutes: { type: "integer" },
              client: { type: "string" },
              color: { type: "string" },
              zoomLink: { type: "string", nullable: true },
              status: { type: "string", enum: ["SCHEDULED", "COMPLETED", "CANCELLED"] },
              therapistId: { type: "string", format: "uuid" },
              patientId: { type: "string", format: "uuid" },
            },
          },
          Conversation: {
            type: "object",
            properties: {
              id: { type: "string", format: "uuid" },
              name: { type: "string" },
              avatar: { type: "string" },
              avatarColor: { type: "string" },
              image: { type: "string", nullable: true },
              lastMessage: { type: "string" },
              time: { type: "string" },
              unread: { type: "integer" },
              active: { type: "boolean" },
            },
          },
          Message: {
            type: "object",
            properties: {
              id: { type: "string", format: "uuid" },
              sender: { type: "string" },
              message: { type: "string" },
              time: { type: "string" },
              isMe: { type: "boolean" },
              avatar: { type: "string" },
              avatarColor: { type: "string" },
              reactions: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    emoji: { type: "string" },
                    count: { type: "integer" },
                  },
                },
              },
            },
          },
          AdminStats: {
            type: "object",
            properties: {
              overview: {
                type: "object",
                properties: {
                  totalUsers: { type: "integer" },
                  therapistCount: { type: "integer" },
                  adminCount: { type: "integer" },
                  patientCount: { type: "integer" },
                  newUsersThisWeek: { type: "integer" },
                  activeSessions: { type: "integer" },
                },
              },
              recentUsers: {
                type: "array",
                items: { $ref: "#/components/schemas/User" },
              },
              authMethods: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    method: { type: "string" },
                    count: { type: "integer" },
                  },
                },
              },
              userGrowth: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    date: { type: "string", format: "date" },
                    users: { type: "integer" },
                  },
                },
              },
            },
          },
        },
      },
      security: [{ sessionAuth: [] }],
      paths: {
        "/api/health": {
          get: {
            tags: ["Health"],
            summary: "Health check",
            description: "Returns health status of the application. No authentication required.",
            security: [],
            responses: {
              "200": {
                description: "Service is healthy",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        status: { type: "string", example: "ok" },
                        timestamp: { type: "string", format: "date-time" },
                      },
                    },
                  },
                },
              },
            },
          },
        },
        "/api/auth/{...all}": {
          get: {
            tags: ["Auth"],
            summary: "Better Auth endpoints",
            description: "All authentication is handled by Better Auth. See Better Auth documentation for available endpoints: /api/auth/sign-in, /api/auth/sign-up, /api/auth/sign-out, etc.",
            security: [],
            parameters: [
              {
                name: "...all",
                in: "path",
                required: true,
                schema: { type: "string" },
                description: "Auth action (sign-in, sign-up, sign-out, callback, etc.)",
              },
            ],
            responses: {
              "200": { description: "Auth response varies by endpoint" },
            },
          },
          post: {
            tags: ["Auth"],
            summary: "Better Auth endpoints",
            description: "POST endpoints for sign-in, sign-up, etc.",
            security: [],
            parameters: [
              {
                name: "...all",
                in: "path",
                required: true,
                schema: { type: "string" },
              },
            ],
            responses: {
              "200": { description: "Auth response varies by endpoint" },
            },
          },
        },
        "/api/user/profile": {
          get: {
            tags: ["User"],
            summary: "Get user profile",
            description: "Returns the authenticated user's profile information",
            responses: {
              "200": {
                description: "User profile",
                content: {
                  "application/json": {
                    schema: { $ref: "#/components/schemas/User" },
                  },
                },
              },
              "401": {
                description: "Unauthorized",
                content: {
                  "application/json": {
                    schema: { $ref: "#/components/schemas/Error" },
                  },
                },
              },
            },
          },
          patch: {
            tags: ["User"],
            summary: "Update user profile",
            description: "Updates the authenticated user's profile (firstname, lastname)",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      firstname: { type: "string" },
                      lastname: { type: "string" },
                    },
                  },
                },
              },
            },
            responses: {
              "200": { description: "Profile updated successfully" },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/user/role": {
          post: {
            tags: ["User"],
            summary: "Set user role",
            description: "Sets the user's role (PATIENT or THERAPIST). Can only be set once.",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      role: { type: "string", enum: ["PATIENT", "THERAPIST"] },
                    },
                    required: ["role"],
                  },
                },
              },
            },
            responses: {
              "200": { description: "Role set successfully" },
              "400": { description: "Invalid role or role already set" },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/user/password": {
          post: {
            tags: ["User"],
            summary: "Change password",
            description: "Changes the authenticated user's password",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      currentPassword: { type: "string" },
                      newPassword: { type: "string", minLength: 8 },
                    },
                    required: ["currentPassword", "newPassword"],
                  },
                },
              },
            },
            responses: {
              "200": { description: "Password changed successfully" },
              "400": { description: "Current password incorrect" },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/user/settings": {
          get: {
            tags: ["User"],
            summary: "Get user settings",
            description: "Returns user settings including 2FA status",
            responses: {
              "200": { description: "User settings" },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/user/connections": {
          get: {
            tags: ["User"],
            summary: "Get user connections",
            description: "Returns therapist-patient connections for the authenticated user",
            responses: {
              "200": {
                description: "List of connections",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        connections: {
                          type: "array",
                          items: { $ref: "#/components/schemas/User" },
                        },
                      },
                    },
                  },
                },
              },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/user/{userId}": {
          get: {
            tags: ["User"],
            summary: "Get user by ID",
            description: "Returns public profile information for a specific user",
            parameters: [
              {
                name: "userId",
                in: "path",
                required: true,
                schema: { type: "string", format: "uuid" },
              },
            ],
            responses: {
              "200": {
                description: "User profile",
                content: {
                  "application/json": {
                    schema: { $ref: "#/components/schemas/User" },
                  },
                },
              },
              "404": { description: "User not found" },
            },
          },
        },
        "/api/appointments": {
          get: {
            tags: ["Appointments"],
            summary: "Get appointments",
            description: "Returns appointments for the authenticated user for the current week (or offset week)",
            parameters: [
              {
                name: "weekOffset",
                in: "query",
                schema: { type: "integer", default: 0 },
                description: "Week offset from current week (negative for past, positive for future)",
              },
            ],
            responses: {
              "200": {
                description: "List of appointments",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        appointments: {
                          type: "array",
                          items: { $ref: "#/components/schemas/Appointment" },
                        },
                      },
                    },
                  },
                },
              },
              "401": { description: "Unauthorized" },
            },
          },
          post: {
            tags: ["Appointments"],
            summary: "Create appointment",
            description: "Creates a new appointment between therapist and patient",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      startAt: { type: "string", format: "date-time" },
                      endAt: { type: "string", format: "date-time" },
                      participantId: { type: "string", format: "uuid", description: "ID of the other participant" },
                    },
                    required: ["startAt", "endAt", "participantId"],
                  },
                },
              },
            },
            responses: {
              "200": {
                description: "Appointment created",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        appointmentId: { type: "string", format: "uuid" },
                      },
                    },
                  },
                },
              },
              "400": { description: "Invalid request" },
              "401": { description: "Unauthorized" },
              "404": { description: "Participant not found" },
            },
          },
          patch: {
            tags: ["Appointments"],
            summary: "Update appointment",
            description: "Updates an existing appointment. Only the therapist can edit.",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      appointmentId: { type: "string", format: "uuid" },
                      startAt: { type: "string", format: "date-time" },
                      endAt: { type: "string", format: "date-time" },
                    },
                    required: ["appointmentId", "startAt", "endAt"],
                  },
                },
              },
            },
            responses: {
              "200": { description: "Appointment updated" },
              "403": { description: "Only therapist can edit" },
              "404": { description: "Appointment not found" },
            },
          },
          delete: {
            tags: ["Appointments"],
            summary: "Cancel appointment",
            description: "Cancels an appointment. Only the patient can cancel.",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      appointmentId: { type: "string", format: "uuid" },
                    },
                    required: ["appointmentId"],
                  },
                },
              },
            },
            responses: {
              "200": { description: "Appointment cancelled" },
              "403": { description: "Only patient can cancel" },
              "404": { description: "Appointment not found" },
            },
          },
        },
        "/api/messages": {
          get: {
            tags: ["Messages"],
            summary: "Get conversations and messages",
            description: "Returns user's conversations and messages for the active conversation",
            parameters: [
              {
                name: "conversationId",
                in: "query",
                schema: { type: "string", format: "uuid" },
                description: "Specific conversation to load messages for",
              },
            ],
            responses: {
              "200": {
                description: "Conversations and messages",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        conversations: {
                          type: "array",
                          items: { $ref: "#/components/schemas/Conversation" },
                        },
                        messages: {
                          type: "array",
                          items: { $ref: "#/components/schemas/Message" },
                        },
                      },
                    },
                  },
                },
              },
              "401": { description: "Unauthorized" },
            },
          },
          post: {
            tags: ["Messages"],
            summary: "Send message or start conversation",
            description: "Sends a message to an existing conversation or starts a new one",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      conversationId: { type: "string", format: "uuid", description: "Existing conversation ID" },
                      recipientId: { type: "string", format: "uuid", description: "Recipient user ID (creates new conversation)" },
                      message: { type: "string" },
                      startOnly: { type: "boolean", description: "If true, only creates/finds conversation without sending message" },
                    },
                  },
                },
              },
            },
            responses: {
              "200": {
                description: "Message sent or conversation created",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        conversation: { $ref: "#/components/schemas/Conversation" },
                        message: { $ref: "#/components/schemas/Message" },
                      },
                    },
                  },
                },
              },
              "400": { description: "Message required (unless startOnly)" },
              "401": { description: "Unauthorized" },
              "404": { description: "Recipient not found" },
            },
          },
          put: {
            tags: ["Messages"],
            summary: "Toggle message reaction",
            description: "Adds or removes an emoji reaction on a message",
            requestBody: {
              required: true,
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      messageId: { type: "string", format: "uuid" },
                      emoji: { type: "string" },
                    },
                    required: ["messageId", "emoji"],
                  },
                },
              },
            },
            responses: {
              "200": {
                description: "Reaction toggled",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        reactions: {
                          type: "array",
                          items: {
                            type: "object",
                            properties: {
                              emoji: { type: "string" },
                              count: { type: "integer" },
                            },
                          },
                        },
                      },
                    },
                  },
                },
              },
              "404": { description: "Message not found" },
            },
          },
        },
        "/api/therapists/available": {
          get: {
            tags: ["Therapists"],
            summary: "Get available therapists",
            description: "Returns list of available therapists for patient discovery",
            responses: {
              "200": {
                description: "List of available therapists",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        therapists: {
                          type: "array",
                          items: { $ref: "#/components/schemas/User" },
                        },
                      },
                    },
                  },
                },
              },
              "401": { description: "Unauthorized" },
            },
          },
        },
        "/api/upload": {
          post: {
            tags: ["Files"],
            summary: "Upload file",
            description: "Uploads a file to cloud storage (GCS in production)",
            requestBody: {
              required: true,
              content: {
                "multipart/form-data": {
                  schema: {
                    type: "object",
                    properties: {
                      file: {
                        type: "string",
                        format: "binary",
                        description: "File to upload",
                      },
                    },
                    required: ["file"],
                  },
                },
              },
            },
            responses: {
              "200": {
                description: "File uploaded successfully",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        url: { type: "string", format: "uri" },
                        pathname: { type: "string" },
                      },
                    },
                  },
                },
              },
              "400": { description: "No file provided" },
              "401": { description: "Unauthorized" },
              "500": { description: "Upload misconfigured" },
            },
          },
        },
        "/api/admin/stats": {
          get: {
            tags: ["Admin"],
            summary: "Get admin statistics",
            description: "Returns platform statistics. Requires administrator role.",
            responses: {
              "200": {
                description: "Admin statistics",
                content: {
                  "application/json": {
                    schema: { $ref: "#/components/schemas/AdminStats" },
                  },
                },
              },
              "401": { description: "Unauthorized" },
              "403": { description: "Forbidden - Admin only" },
            },
          },
        },
        "/api/admin/grafana": {
          get: {
            tags: ["Admin"],
            summary: "Grafana proxy",
            description: "Proxies requests to Grafana dashboard. Requires administrator role.",
            responses: {
              "200": { description: "Grafana dashboard data" },
              "401": { description: "Unauthorized" },
              "403": { description: "Forbidden - Admin only" },
            },
          },
        },
      },
    },
  });
  return spec;
};
