/**
 * Unit Tests for ML Inference Client
 * Tests the mental health classification utilities
 */

import { describe, it, expect } from "vitest";
import {
  isHighRisk,
  getRiskLevel,
  MENTAL_HEALTH_LABELS,
  type PredictionResult,
  type MentalHealthLabel,
} from "@/lib/ml-inference";

describe("ML Inference Client", () => {
  describe("MENTAL_HEALTH_LABELS", () => {
    it("should contain 7 classification labels", () => {
      expect(MENTAL_HEALTH_LABELS).toHaveLength(7);
    });

    it("should include expected labels", () => {
      expect(MENTAL_HEALTH_LABELS).toContain("Anxiety");
      expect(MENTAL_HEALTH_LABELS).toContain("Depression");
      expect(MENTAL_HEALTH_LABELS).toContain("Suicidal");
      expect(MENTAL_HEALTH_LABELS).toContain("Stress");
      expect(MENTAL_HEALTH_LABELS).toContain("Bipolar");
      expect(MENTAL_HEALTH_LABELS).toContain("Personality disorder");
      expect(MENTAL_HEALTH_LABELS).toContain("Normal");
    });
  });

  describe("isHighRisk()", () => {
    it("should return true for Suicidal with high confidence", () => {
      const prediction: PredictionResult = {
        label: "Suicidal",
        confidence: 0.85,
        riskLevel: "high",
      };
      expect(isHighRisk(prediction)).toBe(true);
    });

    it("should return true for Depression with high confidence", () => {
      const prediction: PredictionResult = {
        label: "Depression",
        confidence: 0.9,
        riskLevel: "high",
      };
      expect(isHighRisk(prediction)).toBe(true);
    });

    it("should return false for high risk label with low confidence", () => {
      const prediction: PredictionResult = {
        label: "Suicidal",
        confidence: 0.5,
        riskLevel: "medium",
      };
      expect(isHighRisk(prediction)).toBe(false);
    });

    it("should return false for Normal label", () => {
      const prediction: PredictionResult = {
        label: "Normal",
        confidence: 0.95,
        riskLevel: "normal",
      };
      expect(isHighRisk(prediction)).toBe(false);
    });

    it("should return false for Anxiety even with high confidence", () => {
      const prediction: PredictionResult = {
        label: "Anxiety",
        confidence: 0.9,
        riskLevel: "medium",
      };
      expect(isHighRisk(prediction)).toBe(false);
    });
  });

  describe("getRiskLevel()", () => {
    it("should return 'high' for Suicidal with confidence > 0.5", () => {
      const prediction: PredictionResult = {
        label: "Suicidal",
        confidence: 0.6,
        riskLevel: "high",
      };
      expect(getRiskLevel(prediction)).toBe("high");
    });

    it("should return 'high' for Depression with confidence > 0.7", () => {
      const prediction: PredictionResult = {
        label: "Depression",
        confidence: 0.8,
        riskLevel: "high",
      };
      expect(getRiskLevel(prediction)).toBe("high");
    });

    it("should return 'medium' for Depression with confidence 0.5-0.7", () => {
      const prediction: PredictionResult = {
        label: "Depression",
        confidence: 0.6,
        riskLevel: "medium",
      };
      expect(getRiskLevel(prediction)).toBe("medium");
    });

    it("should return 'medium' for Anxiety with confidence > 0.5", () => {
      const prediction: PredictionResult = {
        label: "Anxiety",
        confidence: 0.7,
        riskLevel: "medium",
      };
      expect(getRiskLevel(prediction)).toBe("medium");
    });

    it("should return 'medium' for Bipolar with confidence > 0.5", () => {
      const prediction: PredictionResult = {
        label: "Bipolar",
        confidence: 0.6,
        riskLevel: "medium",
      };
      expect(getRiskLevel(prediction)).toBe("medium");
    });

    it("should return 'normal' for Normal label", () => {
      const prediction: PredictionResult = {
        label: "Normal",
        confidence: 0.9,
        riskLevel: "normal",
      };
      expect(getRiskLevel(prediction)).toBe("normal");
    });

    it("should return 'low' for Stress with any confidence", () => {
      const prediction: PredictionResult = {
        label: "Stress",
        confidence: 0.8,
        riskLevel: "low",
      };
      expect(getRiskLevel(prediction)).toBe("low");
    });

    it("should return 'low' for Personality disorder with low confidence", () => {
      const prediction: PredictionResult = {
        label: "Personality disorder",
        confidence: 0.4,
        riskLevel: "low",
      };
      expect(getRiskLevel(prediction)).toBe("low");
    });
  });

  describe("Type Safety", () => {
    it("should enforce valid MentalHealthLabel types", () => {
      const validLabels: MentalHealthLabel[] = [
        "Anxiety",
        "Depression",
        "Suicidal",
        "Stress",
        "Bipolar",
        "Personality disorder",
        "Normal",
      ];

      validLabels.forEach((label) => {
        const prediction: PredictionResult = {
          label,
          confidence: 0.5,
          riskLevel: "low",
        };
        expect(prediction.label).toBe(label);
      });
    });
  });
});
