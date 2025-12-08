# Machine Learning & LLM Documentation

## Overview
Document the mental health multi-task classification system built on XLM-RoBERTa Large with 5 prediction heads, trained on Google Colab A100 GPU.

## 1. Model Architecture

### Base Model: XLM-RoBERTa Large
- **Model Name**: `xlm-roberta-large` from HuggingFace
- **Parameters**: 550M total parameters
- **Architecture**: 24-layer transformer with 1024 hidden dimensions, 16 attention heads
- **Training**: Pre-trained on 2.5TB of CommonCrawl data in 100 languages
- **Context Window**: 512 tokens
- **Use Case**: Multilingual text understanding, sentiment analysis, classification tasks

### Multi-Task Learning Architecture
Custom wrapper (`MultiTaskModel`) with 5 specialized prediction heads:

1. **Sentiment Analysis** (Regression)
   - Output: Continuous score from -1 (negative) to +1 (positive)
   - Loss: MSE (Mean Squared Error)
   - Use case: Measure emotional tone in therapy conversations

2. **Family History** (Binary Classification)
   - Output: Binary (0 = no family history, 1 = family history present)
   - Loss: BCEWithLogitsLoss with pos_weight for class imbalance
   - Use case: Identify mentions of family mental health history

3. **Trauma Indicators** (Regression)
   - Output: Scale 0-7 (trauma severity/presence)
   - Loss: MSE
   - Use case: Detect trauma-related language patterns

4. **Social Isolation** (Regression)
   - Output: Scale 0-4 (isolation level)
   - Loss: MSE
   - Use case: Quantify expressions of loneliness and disconnection

5. **Support System** (Regression)
   - Output: Scale 0-4 (support network strength)
   - Loss: MSE
   - Use case: Assess mentions of social support resources

## 2. Training Configuration

### High-Performance Computing
- **Hardware**: Google Colab A100 GPU (40GB VRAM)
- **Precision**: Mixed precision (FP16) with automatic mixed precision (AMP)
- **Gradient Scaling**: CUDA gradient scaler for numerical stability

### Training Strategy: Transfer Learning
- **Frozen Backbone**: XLM-RoBERTa parameters frozen (550M params)
- **Trainable Heads**: Only classification heads trained (~5K params, <1% of total)
- **Rationale**: Faster training, reduced compute, leverages pre-trained language understanding

### Hyperparameters
- **Epochs**: 5
- **Batch Size**: 32
- **Learning Rate**: 1e-3 (higher for head-only training)
- **LR Schedule**: Exponential decay (0.5x per epoch)
- **Optimizer**: AdamW with weight decay
- **Max Sequence Length**: 256 tokens
- **Dropout**: 0.1

### Data Split
- **Training**: 70% of dataset
- **Validation**: 15% (for early stopping)
- **Test**: 15% (final evaluation)

### Loss Function
- **Combined Loss**: Sum of all 5 head losses
- **MSE Loss**: For regression tasks (sentiment, trauma, isolation, support)
- **BCE Loss**: For binary classification (family_history) with computed pos_weight for class imbalance

## 3. Dataset & Features

### Data Source
Mental health counseling conversation dataset with labeled indicators:
- Sentiment scores
- Family history flags
- Trauma severity ratings
- Social isolation levels
- Support system assessments

### Preprocessing
- Tokenization: SentencePiece BPE (from XLM-RoBERTa)
- Truncation: 256 tokens
- Padding: Dynamic per batch
- Normalization: Target scaling to expected ranges

### Class Imbalance Handling
- Family history positive class underrepresented
- Solution: BCEWithLogitsLoss with pos_weight = (neg_count / pos_count)
- Typical weight: ~2-3x for positive class

## 4. Model Training Pipeline

### Local Training
```bash
python ml-training/train.py \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --freeze-backbone true \
  --output-dir ./trained_model
```

### Vertex AI Training
- **Environment Variables**:
  - `AIP_MODEL_DIR`: Output directory for trained model
  - `AIP_TENSORBOARD_LOG_DIR`: TensorBoard logs
  - `CLOUD_ML_PROJECT_ID`: GCP project ID
- **Container**: `ml-training/Dockerfile.vertex`
- **Custom training job** with A100 or V100 GPUs

### Training Outputs
- **Model Weights**: `model_v{timestamp}.pt` (PyTorch state dict)
- **Tokenizer**: HuggingFace tokenizer config files
- **Metrics**: `metrics.json` with training/validation/test losses per epoch
- **Versioning**: Timestamp-based versions (e.g., `v20251208_143022`)

## 5. Model Deployment

### Inference Server
- **Framework**: FastAPI + PyTorch
- **Container**: `ml-training/Dockerfile.inference`
- **Endpoints**:
  - `POST /predict`: Single text inference
  - `POST /predict_batch`: Batch inference (up to 32 samples)
  - `GET /health`: Health check
- **Response Format**: JSON with all 5 prediction heads

### Vertex AI Endpoint
- **Managed Deployment**: Vertex AI Prediction service
- **Auto-scaling**: Min 1, max 5 nodes
- **GPU**: Optional (NVIDIA T4 for faster inference)
- **Monitoring**: Integrated with Cloud Logging and Monitoring

### Integration with Bloom Platform
- **Message Analysis**: Real-time inference on therapy messages
- **API Route**: `src/app/api/ml/analyze/route.ts`
- **Caching**: Redis cache for repeated queries (TTL: 1 hour)
- **Async Processing**: Non-blocking inference with fallback

## 6. Performance Metrics

### Training Performance
- **Training Time**: ~15-20 minutes on A100 (5 epochs, 32 batch size)
- **Validation Loss**: Tracked per epoch for early stopping
- **Test Loss**: Final evaluation on held-out test set
- **Per-Head Losses**: Individual loss tracking for each prediction task

### Inference Performance
- **Latency**: <50ms per sample (A100), <200ms (CPU)
- **Throughput**: 50-100 requests/second (GPU deployment)
- **Batch Processing**: Up to 32 samples in parallel

### Model Quality (Example Metrics)
- **Sentiment MAE**: ~0.15 (on -1 to 1 scale)
- **Family History F1**: ~0.82
- **Trauma MAE**: ~0.8 (on 0-7 scale)
- **Isolation MAE**: ~0.5 (on 0-4 scale)
- **Support MAE**: ~0.6 (on 0-4 scale)

## 7. Model Versioning & MLOps

### Version Control
- **Model Registry**: GCS bucket `gs://bloom-health-ml-models/`
- **Naming Convention**: `v{YYYYMMDD}_{HHMMSS}`
- **Latest Symlink**: `latest/` points to current production model
- **Rollback**: Previous versions retained for 90 days

### CI/CD Integration
- **Training Trigger**: Manual or scheduled (weekly retraining)
- **Automated Testing**: Validation loss threshold check before deployment
- **Canary Deployment**: 10% traffic to new model for 24 hours
- **Monitoring**: Prediction drift detection via Vertex AI Model Monitoring

### Experimentation
- **Notebook**: `Final_2.ipynb` (Google Colab)
- **Experiment Tracking**: Manual metrics logging to `metrics.json`
- **A/B Testing**: Version-based routing in API gateway

## 8. LLM Features Used

### Pre-trained Capabilities (XLM-RoBERTa)
- **Multilingual Understanding**: 100 languages (English primary for Bloom)
- **Contextual Embeddings**: Bidirectional attention for nuanced text understanding
- **Transfer Learning**: Pre-trained on CommonCrawl eliminates need for large labeled dataset

### Fine-Tuning Strategy
- **Task**: Multi-task learning for mental health text analysis
- **Approach**: Head-only training (freeze backbone, train 5 regression/classification heads)
- **Benefit**: Faster training (20 min vs. hours), lower compute cost, reduced overfitting risk

### Feature Extraction
- **[CLS] Token**: First token embedding used as sentence representation
- **Pooling**: No additional pooling (direct [CLS] → head projection)
- **Dropout**: 0.1 dropout on [CLS] embedding before heads

## 9. Ethical Considerations & Limitations

### Privacy & Security
- **HIPAA Compliance**: Model never stores patient data, only processes ephemeral requests
- **De-identification**: Training data anonymized before model training
- **Opt-in**: Users must consent to ML-assisted analysis

### Limitations
- **English-only**: Despite multilingual backbone, training data is English-focused
- **Bias Risk**: Model reflects biases in training data (mental health counseling corpus)
- **Not Diagnostic**: Predictions are assistive tools, not clinical diagnoses
- **Human-in-Loop**: All ML outputs reviewed by licensed therapists

### Monitoring & Auditing
- **Prediction Logging**: All inferences logged (without PII) for audit
- **Drift Detection**: Monthly review of prediction distributions
- **Feedback Loop**: Therapist corrections used to identify model weaknesses

## 10. Future Enhancements

### Model Improvements
- **Fine-tune Backbone**: Unfreeze last 4 layers for better task adaptation
- **Multi-modal**: Add audio/video analysis for telehealth sessions
- **Personalization**: User-specific fine-tuning with consent

### Infrastructure
- **Edge Deployment**: TensorFlow Lite or ONNX for client-side inference (privacy)
- **Active Learning**: Continuous learning from therapist corrections
- **Ensemble Models**: Combine multiple models for robustness

### Explainability
- **Attention Visualization**: Show which words influenced predictions
- **SHAP Values**: Quantify feature importance
- **Confidence Scores**: Uncertainty quantification for predictions

## References

- **Base Model**: [XLM-RoBERTa Large on HuggingFace](https://huggingface.co/xlm-roberta-large)
- **Training Notebook**: `Final_2.ipynb` (Google Colab with A100)
- **Training Script**: `ml-training/train.py`
- **Model Definition**: `ml-training/model.py`
- **Inference Server**: `ml-training/inference.py`
- **Deployment**: `ml-training/Dockerfile.vertex`, `ml-training/Dockerfile.inference`

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-08  
**Authors**: Bloom ML Team
