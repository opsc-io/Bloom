/**
 * ML Inference Client for Mental Health Classification
 * Supports both Vertex AI Endpoint and internal K8s service
 */

// Mental health classification labels
export const MENTAL_HEALTH_LABELS = [
  "Anxiety",
  "Depression",
  "Suicidal",
  "Stress",
  "Bipolar",
  "Personality disorder",
  "Normal",
] as const;

export type MentalHealthLabel = (typeof MENTAL_HEALTH_LABELS)[number];

export interface PsychometricProfile {
  sentiment: number; // -1 to 1
  trauma: number; // 0 to 7
  isolation: number; // 0 to 4
  support: number; // 0 to ~4
  familyHistoryProb: number; // 0 to 1
}

export interface PredictionResult {
  label: MentalHealthLabel;
  confidence: number;
  riskLevel: "high" | "medium" | "low" | "normal";
  allScores?: Record<MentalHealthLabel, number>;
  psychometrics?: PsychometricProfile;
}

export interface MLHealthStatus {
  status: "healthy" | "unhealthy";
  modelLoaded: boolean;
  device: string;
  labels: string[];
}

// Configuration - supports both Vertex AI and direct service
const VERTEX_AI_ENDPOINT = process.env.VERTEX_AI_ENDPOINT;
const VERTEX_AI_PROJECT = process.env.VERTEX_AI_PROJECT || "project-4fc52960-1177-49ec-a6f";
const VERTEX_AI_LOCATION = process.env.VERTEX_AI_LOCATION || "us-central1";

// Fallback to internal K8s service or mock
const ML_INFERENCE_URL =
  process.env.ML_INFERENCE_URL || "http://ml-inference:8080";

// Use mock ML when no real service is configured
export const USE_MOCK_ML =
  process.env.USE_MOCK_ML === "true" ||
  (!VERTEX_AI_ENDPOINT && !process.env.ML_INFERENCE_URL);

/**
 * Get Google Cloud auth token for Vertex AI
 * Uses Application Default Credentials (ADC) on GKE, or gcloud CLI locally
 */
async function getGoogleAuthToken(): Promise<string> {
  // On GKE with Workload Identity, fetch from metadata server
  try {
    const response = await fetch(
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
      { headers: { "Metadata-Flavor": "Google" } }
    );
    if (response.ok) {
      const data = await response.json();
      return data.access_token;
    }
  } catch {
    // Not running on GCP - try local gcloud CLI
  }

  // Local development: use gcloud CLI to get access token
  try {
    const { execSync } = await import("child_process");
    const token = execSync("gcloud auth print-access-token", {
      encoding: "utf-8",
      timeout: 5000,
    }).trim();
    if (token) {
      return token;
    }
  } catch {
    // gcloud CLI not available or not authenticated
  }

  throw new Error("Unable to get Google Cloud auth token. Run 'gcloud auth login' for local development.");
}

/**
 * Mock analyzer - returns classification based on text analysis
 * Used when ML model isn't ready yet
 */
function mockAnalyzeText(text: string): PredictionResult {
  const lowerText = text.toLowerCase();

  // Simple keyword-based mock analysis
  let label: MentalHealthLabel = "Normal";
  let confidence = 0.65;
  let riskLevel: "high" | "medium" | "low" | "normal" = "normal";

  // High risk keywords
  if (lowerText.includes("suicid") || lowerText.includes("kill myself") || lowerText.includes("end my life")) {
    label = "Suicidal";
    confidence = 0.85;
    riskLevel = "high";
  } else if (lowerText.includes("depress") || lowerText.includes("hopeless") || lowerText.includes("worthless")) {
    label = "Depression";
    confidence = 0.75;
    riskLevel = "medium";
  } else if (lowerText.includes("anxious") || lowerText.includes("panic") || lowerText.includes("worried")) {
    label = "Anxiety";
    confidence = 0.70;
    riskLevel = "medium";
  } else if (lowerText.includes("stress") || lowerText.includes("overwhelm") || lowerText.includes("pressure")) {
    label = "Stress";
    confidence = 0.70;
    riskLevel = "low";
  }

  // Generate mock psychometric scores
  const psychometrics: PsychometricProfile = {
    sentiment: label === "Normal" ? 0.3 : -0.4,
    trauma: ["Suicidal", "Depression"].includes(label) ? 4.5 : 1.5,
    isolation: ["Suicidal", "Depression"].includes(label) ? 2.8 : 1.0,
    support: label === "Normal" ? 0.8 : 0.3,
    familyHistoryProb: 0.2,
  };

  // Generate mock scores for all labels
  const allScores = MENTAL_HEALTH_LABELS.reduce(
    (acc, l) => ({
      ...acc,
      [l]: l === label ? confidence : Math.random() * 0.3,
    }),
    {} as Record<MentalHealthLabel, number>
  );

  return { label, confidence, riskLevel, allScores, psychometrics };
}

/**
 * Determine label from psychometric scores
 * This provides client-side label determination based on the actual model output ranges
 */
function determineLabelFromPsychometrics(
  psychometrics: PsychometricProfile
): { label: MentalHealthLabel; confidence: number } {
  const { sentiment, trauma, isolation, support } = psychometrics;

  // Very negative sentiment with any trauma indicator suggests serious conditions
  if (sentiment < -0.5) {
    if (trauma > 0.6 || isolation > 0.5) {
      return { label: "Suicidal", confidence: Math.min(0.95, 0.7 + Math.abs(sentiment) * 0.2) };
    }
    return { label: "Depression", confidence: Math.min(0.90, 0.6 + Math.abs(sentiment) * 0.3) };
  }

  // Moderately negative sentiment
  if (sentiment < -0.2) {
    if (trauma > 0.5 && isolation > 0.4) {
      return { label: "Depression", confidence: Math.min(0.85, 0.5 + trauma * 0.3) };
    }
    if (trauma > 0.4 || isolation > 0.4) {
      return { label: "Anxiety", confidence: Math.min(0.85, 0.5 + trauma * 0.3) };
    }
    return { label: "Stress", confidence: Math.min(0.80, 0.5 + isolation * 0.4) };
  }

  // Slightly negative sentiment (mild stress/anxiety)
  if (sentiment < 0) {
    if (trauma > 0.5 || isolation > 0.5) {
      return { label: "Anxiety", confidence: Math.min(0.75, 0.45 + trauma * 0.3) };
    }
    return { label: "Stress", confidence: Math.min(0.70, 0.4 + isolation * 0.4) };
  }

  // Positive sentiment - check if it's genuine or masking
  if (sentiment > 0.3) {
    // High positive sentiment with low trauma/isolation = Normal
    if (trauma < 0.4 && isolation < 0.4) {
      return { label: "Normal", confidence: Math.min(0.90, 0.6 + sentiment * 0.3) };
    }
    // Positive sentiment but elevated trauma could be bipolar or masking
    if (trauma > 0.5) {
      return { label: "Bipolar", confidence: Math.min(0.70, 0.4 + trauma * 0.3) };
    }
    return { label: "Normal", confidence: Math.min(0.80, 0.5 + sentiment * 0.3) };
  }

  // Neutral sentiment (0 to 0.3) - depends on other factors
  if (trauma > 0.5 || isolation > 0.5) {
    return { label: "Stress", confidence: Math.min(0.75, 0.4 + trauma * 0.3) };
  }

  // Default to Normal for neutral/positive with low indicators
  return { label: "Normal", confidence: Math.min(0.70, 0.5 + sentiment * 0.2) };
}

/**
 * Call Vertex AI Endpoint for prediction
 */
async function callVertexAI(text: string): Promise<PredictionResult> {
  const token = await getGoogleAuthToken();
  const endpoint = `https://${VERTEX_AI_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEX_AI_PROJECT}/locations/${VERTEX_AI_LOCATION}/endpoints/${VERTEX_AI_ENDPOINT}:predict`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      instances: [{ text, return_all_scores: true }],
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Vertex AI prediction failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  const prediction = data.predictions?.[0]?.prediction || data.predictions?.[0];

  // Extract psychometrics
  const psychometrics: PsychometricProfile | undefined = prediction.psychometrics ? {
    sentiment: prediction.psychometrics.sentiment,
    trauma: prediction.psychometrics.trauma,
    isolation: prediction.psychometrics.isolation,
    support: prediction.psychometrics.support,
    familyHistoryProb: prediction.psychometrics.family_history_prob,
  } : undefined;

  // Re-calculate label from psychometrics (client-side) for better accuracy
  let label = prediction.label as MentalHealthLabel;
  let confidence = prediction.confidence;

  if (psychometrics) {
    const recalculated = determineLabelFromPsychometrics(psychometrics);
    label = recalculated.label;
    confidence = recalculated.confidence;
    console.log(`[ML] Recalculated label: ${label} (was: ${prediction.label}), confidence: ${confidence.toFixed(2)}`);
  }

  return {
    label,
    confidence,
    riskLevel: getRiskLevelFromPrediction(label, confidence),
    psychometrics,
    allScores: prediction.all_scores,
  };
}

/**
 * Call internal ML service for prediction
 */
async function callMLService(text: string, returnAllScores: boolean): Promise<PredictionResult> {
  const response = await fetch(`${ML_INFERENCE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      return_all_scores: returnAllScores,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`ML inference failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return {
    label: data.prediction.label as MentalHealthLabel,
    confidence: data.prediction.confidence,
    riskLevel: data.prediction.risk_level || getRiskLevelFromPrediction(data.prediction.label, data.prediction.confidence),
    psychometrics: data.prediction.psychometrics ? {
      sentiment: data.prediction.psychometrics.sentiment,
      trauma: data.prediction.psychometrics.trauma,
      isolation: data.prediction.psychometrics.isolation,
      support: data.prediction.psychometrics.support,
      familyHistoryProb: data.prediction.psychometrics.family_history_prob,
    } : undefined,
    allScores: data.prediction.all_scores,
  };
}

/**
 * Analyze a single text for mental health indicators
 * Uses Vertex AI > Internal Service > Mock (in that priority order)
 */
export async function analyzeText(
  text: string,
  returnAllScores = false
): Promise<PredictionResult> {
  // Use mock when no ML service is configured
  if (USE_MOCK_ML) {
    console.log("[ML] Using mock analyzer (USE_MOCK_ML=true)");
    return mockAnalyzeText(text);
  }

  // Try Vertex AI first if configured
  if (VERTEX_AI_ENDPOINT) {
    try {
      console.log("[ML] Using Vertex AI endpoint");
      return await callVertexAI(text);
    } catch (error) {
      console.warn("[ML] Vertex AI failed, falling back:", error);
    }
  }

  // Fall back to internal service
  console.log("[ML] Using internal ML service");
  return await callMLService(text, returnAllScores);
}

/**
 * Analyze multiple texts in a batch
 */
export async function analyzeTexts(
  texts: string[],
  returnAllScores = false
): Promise<PredictionResult[]> {
  // For Vertex AI, process individually (or implement batch)
  if (VERTEX_AI_ENDPOINT && !USE_MOCK_ML) {
    return Promise.all(texts.map(text => analyzeText(text, returnAllScores)));
  }

  // Use mock for all
  if (USE_MOCK_ML) {
    return texts.map(text => mockAnalyzeText(text));
  }

  // Use internal batch endpoint
  const response = await fetch(`${ML_INFERENCE_URL}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      texts,
      return_all_scores: returnAllScores,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`ML batch inference failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.predictions.map(
    (p: { label: string; confidence: number; risk_level?: string; all_scores?: object; psychometrics?: object }) => ({
      label: p.label as MentalHealthLabel,
      confidence: p.confidence,
      riskLevel: p.risk_level || getRiskLevelFromPrediction(p.label as MentalHealthLabel, p.confidence),
      allScores: p.all_scores,
    })
  );
}

/**
 * Check ML service health
 */
export async function checkMLHealth(): Promise<MLHealthStatus> {
  if (USE_MOCK_ML) {
    return {
      status: "healthy",
      modelLoaded: true,
      device: "mock",
      labels: [...MENTAL_HEALTH_LABELS],
    };
  }

  try {
    const response = await fetch(`${ML_INFERENCE_URL}/health`);
    if (!response.ok) {
      return {
        status: "unhealthy",
        modelLoaded: false,
        device: "unknown",
        labels: [],
      };
    }
    const data = await response.json();
    return {
      status: data.status,
      modelLoaded: data.model_loaded,
      device: data.device,
      labels: data.labels,
    };
  } catch {
    return {
      status: "unhealthy",
      modelLoaded: false,
      device: "unknown",
      labels: [],
    };
  }
}

/**
 * Check if a prediction indicates high risk
 */
export function isHighRisk(prediction: PredictionResult): boolean {
  return prediction.riskLevel === "high";
}

/**
 * Get risk level from prediction (for backwards compatibility)
 */
function getRiskLevelFromPrediction(
  label: MentalHealthLabel,
  confidence: number
): "high" | "medium" | "low" | "normal" {
  if (label === "Suicidal" && confidence > 0.5) {
    return "high";
  }
  if (label === "Depression" && confidence > 0.7) {
    return "high";
  }
  if (
    ["Depression", "Anxiety", "Bipolar"].includes(label) &&
    confidence > 0.5
  ) {
    return "medium";
  }
  if (label === "Normal") {
    return "normal";
  }
  return "low";
}

/**
 * Get risk level from prediction (exported for backward compatibility)
 */
export function getRiskLevel(
  prediction: PredictionResult
): "high" | "medium" | "low" | "normal" {
  return prediction.riskLevel;
}
