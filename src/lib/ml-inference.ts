/**
 * ML Inference Client for Mental Health Classification
 * Calls the internal ml-inference service in the cluster
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

export interface PredictionResult {
  label: MentalHealthLabel;
  confidence: number;
  allScores?: Record<MentalHealthLabel, number>;
}

export interface MLHealthStatus {
  status: "healthy" | "unhealthy";
  modelLoaded: boolean;
  device: string;
  labels: string[];
}

// Service URL - internal K8s service or environment override
const ML_INFERENCE_URL =
  process.env.ML_INFERENCE_URL || "http://ml-inference:8080";

/**
 * Analyze a single text for mental health indicators
 */
export async function analyzeText(
  text: string,
  returnAllScores = false
): Promise<PredictionResult> {
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
    label: data.prediction.label,
    confidence: data.prediction.confidence,
    allScores: data.prediction.all_scores,
  };
}

/**
 * Analyze multiple texts in a batch
 */
export async function analyzeTexts(
  texts: string[],
  returnAllScores = false
): Promise<PredictionResult[]> {
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
    (p: { label: string; confidence: number; all_scores?: object }) => ({
      label: p.label as MentalHealthLabel,
      confidence: p.confidence,
      allScores: p.all_scores,
    })
  );
}

/**
 * Check ML service health
 */
export async function checkMLHealth(): Promise<MLHealthStatus> {
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
  const highRiskLabels: MentalHealthLabel[] = ["Suicidal", "Depression"];
  return (
    highRiskLabels.includes(prediction.label) && prediction.confidence > 0.7
  );
}

/**
 * Get risk level from prediction
 */
export function getRiskLevel(
  prediction: PredictionResult
): "high" | "medium" | "low" | "normal" {
  if (prediction.label === "Suicidal" && prediction.confidence > 0.5) {
    return "high";
  }
  if (prediction.label === "Depression" && prediction.confidence > 0.7) {
    return "high";
  }
  if (
    ["Depression", "Anxiety", "Bipolar"].includes(prediction.label) &&
    prediction.confidence > 0.5
  ) {
    return "medium";
  }
  if (prediction.label === "Normal") {
    return "normal";
  }
  return "low";
}
