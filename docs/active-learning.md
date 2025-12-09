# Active Learning Loop

Bloom ships a therapist-in-the-loop feedback loop to continuously improve the mental health classifier.

## Components
- **Feedback capture**: `src/app/api/ml/feedback/route.ts` (POST/GET). Therapists mark predictions correct/incorrect and supply corrected labels + notes.
- **Schema**: `prisma/schema.prisma` model `MLFeedback` (unique per `analysisId`, indexed by therapist and timestamp).
- **Storage & joins**: Feedback rows attach to `MessageAnalysis` and `User` via Prisma relations; see migration `prisma/migrations/20251208090537/`.
- **Stats endpoint**: Same route exposes aggregate accuracy, correction counts, and recent examples for dashboards.

## Training Data Flow
1. **Export** therapist feedback from BigQuery via `ml/pipelines/retrain_pipeline.py::export_therapist_feedback()` (defaults to last 30 days, filters short/null messages).
2. **Combine** with existing labeled corpus in `prepare_training_data()`; shuffles and re-splits 90/10 for train/val.
3. **Train** new heads-only model (`train.py`) using Vertex AI Custom Jobs; metrics persisted to `metrics.json` and GCS.
4. **Compare & gate**: Deploy only if new model beats current by ≥2% (`IMPROVEMENT_THRESHOLD`) on validation loss.

## Operational Notes
- Only therapists can submit corrections; admins/therapists can read stats.
- Corrections must map to the canonical 7 labels (`ML_LABELS` in the route), avoiding noisy free text.
- Feedback export enforces minimum message length and strips rows with missing labels to keep the signal clean.

## Current Status
- Feedback capture + stats API: **implemented**.
- BigQuery export + Vertex retrain pipeline: **implemented** (see `ml/pipelines/retrain_pipeline.py`).
- Automated promotion/canary on better metrics: **pending**; currently a manual deploy after reviewing `metrics.json`.
- UI surfacing of feedback stats: **pending**; server route returns `recentFeedback` payload for a thin dashboard.

## Future Work
- Add reviewer assignment + double-check workflow for high-risk label corrections.
- Weight recent feedback higher when sampling retrain data to reduce staleness.
- Alerting when feedback volume drops below threshold (active learning health).
