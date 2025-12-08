"""
Shared model definition for Mental Health Multi-Task Model.
Used by both training and serving components.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "xlm-roberta-large"


class MultiTaskModel(nn.Module):
    """
    Multi-task model with XLM-RoBERTa backbone and task-specific heads.

    Outputs:
        - sentiment: Emotional valence (-1 to 1)
        - trauma: Trauma indicator level (0 to 7)
        - isolation: Social isolation score (0 to 4)
        - support: Support system strength (0 to 1)
        - family: Family history probability (0 to 1)
    """

    def __init__(self, pretrained_model_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Task-specific heads
        self.head_sentiment = nn.Linear(self.hidden_size, 1)
        self.head_trauma = nn.Linear(self.hidden_size, 1)
        self.head_isolation = nn.Linear(self.hidden_size, 1)
        self.head_support = nn.Linear(self.hidden_size, 1)
        self.head_family = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass through model."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return {
            "sentiment": torch.tanh(self.head_sentiment(pooled)).squeeze(-1),      # [-1, 1]
            "trauma": torch.relu(self.head_trauma(pooled)).squeeze(-1) * 7,        # [0, 7]
            "isolation": torch.relu(self.head_isolation(pooled)).squeeze(-1) * 4,  # [0, 4]
            "support": torch.sigmoid(self.head_support(pooled)).squeeze(-1),       # [0, 1]
            "family": torch.sigmoid(self.head_family(pooled)).squeeze(-1),         # [0, 1]
        }

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": MODEL_NAME,
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(pretrained_model_name=checkpoint.get("model_name", MODEL_NAME))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def predict(self, text: str, tokenizer, device: str = "cpu") -> dict:
        """Convenience method for single text prediction."""
        self.eval()

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self(input_ids, attention_mask)

        return {
            "sentiment": float(outputs["sentiment"].cpu().item()),
            "trauma": float(outputs["trauma"].cpu().item()),
            "isolation": float(outputs["isolation"].cpu().item()),
            "support": float(outputs["support"].cpu().item()),
            "family_history": float(outputs["family"].cpu().item()),
        }
