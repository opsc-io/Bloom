# Testing, Data Validation, and Cross-Validation

## What We Test
- **Unit & integration**: `npm test` runs Vitest suites in `src/__tests__` (auth, health, admin stats, ML inference client). `npm run test:coverage` produces coverage artifacts locally.
- **API surface smoke**: `src/__tests__/api-health.test.ts` exercises the health route to ensure env wiring and Prisma bootstrapping don’t regress.
- **ML client contracts**: `src/__tests__/ml-inference.test.ts` asserts label set, risk thresholds, and type safety for the message analysis helpers.

## Data Validation (training pipeline)
- **Schema + ranges** (enforced in `ml-training/dataset.py`):
  - Required columns: `text`, `label`, `sentiment`, `trauma`, `isolation`, `support`, `family_history`.
  - Observed ranges on load: `sentiment` [-1, 1], `trauma` [0, 7], `isolation` [0, 4], `support` [0, 4], `family_history` ∈ {0,1}.
- **Quality filters**:
  - Drop rows with null/blank text or labels with <2 examples to keep stratified splits stable.
  - Enforce minimum length (`LENGTH(text) >= 50`) before exporting feedback data in `ml/pipelines/retrain_pipeline.py`.
- **Split strategy**:
  - Stratified 70/15/15 train/val/test using `train_test_split` on the label column.
  - Class balance logged per split; runs abort if any label disappears from val/test.

## Cross-Validation Results (latest recorded)
Source: `Final_2.ipynb` two-epoch run on the stratified 70/15/15 split (frozen backbone, heads-only training).

| Epoch | Train Loss | Val Loss | Val Sentiment | Val Trauma | Val Isolation | Val Support | Val Family |
|-------|------------|----------|---------------|------------|---------------|-------------|------------|
| 1 | 2.6523 | 2.1954 | 0.1132 | 0.5882 | 0.3216 | 0.3558 | 0.8165 |
| 2 (best) | 2.3790 | **2.0940** | 0.1123 | 0.5795 | 0.3081 | 0.3497 | 0.7445 |

Notes:
- Backbone frozen; only prediction heads trained. Learning rate decayed 2× after epoch 1.
- Validation loss is the summed per-head loss (4× MSE + 1× weighted BCE). Per-head metrics saved to `metrics.json` when `train.py` runs under Vertex AI/CI.
- Test-set evaluation mirrors validation, executed after restoring the best-val checkpoint in `train.py`.

## How to Reproduce Locally
```bash
cd ml-training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 2 --batch-size 16 --output-dir ./trained_model
```

## CI/Lint Hooks
- `npm run lint` runs ESLint for the Next.js app.
- `npm test` is wired into CI; add new suites under `src/__tests__` to extend regression coverage.***
