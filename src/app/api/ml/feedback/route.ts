import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { auth } from "@/lib/auth";
import { UserRole } from "@/generated/prisma/client";

const ML_LABELS = [
  "Anxiety",
  "Depression",
  "Suicidal",
  "Stress",
  "Bipolar",
  "Personality disorder",
  "Normal",
];

/**
 * POST /api/ml/feedback
 * Submit therapist feedback on ML predictions for active learning
 *
 * Body: {
 *   analysisId: string,     // ID of MessageAnalysis record
 *   isCorrect: boolean,     // Was the prediction correct?
 *   correctedLabel?: string, // If incorrect, the correct label
 *   notes?: string          // Optional notes
 * }
 */
export async function POST(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Only therapists can submit feedback
  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { role: true },
  });

  if (user?.role !== UserRole.THERAPIST) {
    return NextResponse.json(
      { error: "Only therapists can provide ML feedback" },
      { status: 403 }
    );
  }

  try {
    const body = await req.json();
    const { analysisId, isCorrect, correctedLabel, notes } = body;

    // Validate required fields
    if (!analysisId || typeof isCorrect !== "boolean") {
      return NextResponse.json(
        { error: "analysisId and isCorrect are required" },
        { status: 400 }
      );
    }

    // Validate correctedLabel if provided
    if (correctedLabel && !ML_LABELS.includes(correctedLabel)) {
      return NextResponse.json(
        { error: `Invalid label. Must be one of: ${ML_LABELS.join(", ")}` },
        { status: 400 }
      );
    }

    // If incorrect, correctedLabel should be provided
    if (!isCorrect && !correctedLabel) {
      return NextResponse.json(
        { error: "correctedLabel is required when isCorrect is false" },
        { status: 400 }
      );
    }

    // Check analysis exists
    const analysis = await prisma.messageAnalysis.findUnique({
      where: { id: analysisId },
    });

    if (!analysis) {
      return NextResponse.json(
        { error: "Analysis not found" },
        { status: 404 }
      );
    }

    // Create or update feedback
    const feedback = await prisma.mLFeedback.upsert({
      where: { analysisId },
      update: {
        isCorrect,
        correctedLabel: isCorrect ? null : correctedLabel,
        notes,
        therapistId: session.user.id,
      },
      create: {
        analysisId,
        therapistId: session.user.id,
        isCorrect,
        correctedLabel: isCorrect ? null : correctedLabel,
        notes,
      },
    });

    return NextResponse.json({
      success: true,
      feedback: {
        id: feedback.id,
        isCorrect: feedback.isCorrect,
        correctedLabel: feedback.correctedLabel,
      },
    });
  } catch (error) {
    console.error("ML Feedback error:", error);
    return NextResponse.json(
      { error: "Failed to submit feedback" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/ml/feedback
 * Get feedback statistics for active learning monitoring
 */
export async function GET(req: Request) {
  const session = await auth.api.getSession({ headers: req.headers });
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Only admins and therapists can view stats
  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { role: true },
  });

  if (user?.role !== UserRole.THERAPIST && user?.role !== UserRole.ADMINISTRATOR) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  try {
    // Get overall accuracy stats
    const totalFeedback = await prisma.mLFeedback.count();
    const correctCount = await prisma.mLFeedback.count({
      where: { isCorrect: true },
    });

    // Get accuracy by label
    const feedbackByLabel = await prisma.mLFeedback.groupBy({
      by: ["isCorrect"],
      _count: true,
    });

    // Get recent feedback for review
    const recentFeedback = await prisma.mLFeedback.findMany({
      take: 10,
      orderBy: { createdAt: "desc" },
      include: {
        analysis: {
          select: {
            label: true,
            confidence: true,
            riskLevel: true,
            message: {
              select: {
                body: true,
              },
            },
          },
        },
        therapist: {
          select: {
            firstname: true,
            lastname: true,
          },
        },
      },
    });

    // Count corrections by original label
    const correctionsByLabel = await prisma.mLFeedback.groupBy({
      by: ["correctedLabel"],
      where: { isCorrect: false, correctedLabel: { not: null } },
      _count: true,
    });

    return NextResponse.json({
      stats: {
        totalFeedback,
        correctCount,
        incorrectCount: totalFeedback - correctCount,
        accuracy: totalFeedback > 0 ? (correctCount / totalFeedback) * 100 : 0,
      },
      correctionsByLabel: correctionsByLabel.map((c) => ({
        label: c.correctedLabel,
        count: c._count,
      })),
      recentFeedback: recentFeedback.map((f) => ({
        id: f.id,
        isCorrect: f.isCorrect,
        correctedLabel: f.correctedLabel,
        notes: f.notes,
        createdAt: f.createdAt,
        originalPrediction: {
          label: f.analysis.label,
          confidence: f.analysis.confidence,
          riskLevel: f.analysis.riskLevel,
        },
        therapist: `${f.therapist.firstname} ${f.therapist.lastname}`,
        messagePreview: f.analysis.message?.body?.substring(0, 100) || "",
      })),
    });
  } catch (error) {
    console.error("ML Feedback stats error:", error);
    return NextResponse.json(
      { error: "Failed to fetch feedback stats" },
      { status: 500 }
    );
  }
}
