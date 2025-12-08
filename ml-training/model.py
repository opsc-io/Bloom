"""
Multi-Task Mental Health Classification Model
Based on XLM-RoBERTa Large backbone with 5 prediction heads
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import os


# Default model configuration
DEFAULT_MODEL_NAME = "xlm-roberta-large"
HIDDEN_SIZE = 1024  # XLM-RoBERTa Large hidden size


class MultiTaskModel(nn.Module):
    """
    Multi-task model for mental health text analysis.

    Prediction Heads:
    1. Sentiment (Regression: -1 to 1)
    2. Family History (Binary Classification: 0 or 1)
    3. Trauma (Regression: 0 to 7)
    4. Isolation (Regression: 0 to 4)
    5. Support (Regression: scaled 0 to ~4)
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        dropout_rate: float = 0.1,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = pretrained_model
        self.hidden_size = self.backbone.config.hidden_size

        # Freeze backbone for faster training (train only heads)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Prediction heads
        self.head_sentiment = nn.Linear(self.hidden_size, 1)
        self.head_family = nn.Linear(self.hidden_size, 1)  # Outputs logit
        self.head_trauma = nn.Linear(self.hidden_size, 1)
        self.head_isolation = nn.Linear(self.hidden_size, 1)
        self.head_support = nn.Linear(self.hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone and all prediction heads.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with predictions for each head
        """
        # Pass through transformer backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)

        # Pass through each head
        return {
            'sentiment': self.head_sentiment(cls_embedding),
            'family': self.head_family(cls_embedding),
            'trauma': self.head_trauma(cls_embedding),
            'isolation': self.head_isolation(cls_embedding),
            'support': self.head_support(cls_embedding)
        }


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[torch.device] = None
) -> tuple:
    """
    Load the tokenizer and base model.

    Args:
        model_name: HuggingFace model name
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (tokenizer, model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    return tokenizer, base_model, device


def create_multitask_model(
    base_model: nn.Module,
    device: torch.device,
    dropout_rate: float = 0.1,
    freeze_backbone: bool = True
) -> MultiTaskModel:
    """
    Create the multi-task model wrapper.

    Args:
        base_model: Pre-trained transformer model
        device: Target device
        dropout_rate: Dropout probability
        freeze_backbone: If True, freeze backbone and train only heads

    Returns:
        MultiTaskModel on the specified device
    """
    model = MultiTaskModel(base_model, dropout_rate, freeze_backbone)
    return model.to(device)


def save_model(
    model: MultiTaskModel,
    tokenizer: AutoTokenizer,
    save_dir: str,
    model_version: str = "v1"
):
    """
    Save model weights and tokenizer.

    Args:
        model: Trained MultiTaskModel
        tokenizer: Associated tokenizer
        save_dir: Directory to save to
        model_version: Version string for the model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights (just the heads + backbone)
    model_path = os.path.join(save_dir, f"model_{model_version}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'version': model_version
    }, model_path)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    print(f"Model saved to {save_dir}")
    print(f"  - Weights: {model_path}")
    print(f"  - Tokenizer: {save_dir}/")


def load_trained_model(
    save_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[torch.device] = None
) -> tuple:
    """
    Load a trained model from disk.

    Args:
        save_dir: Directory containing saved model
        model_name: Base model name (for backbone)
        device: Target device

    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    # Load base model and create wrapper
    base_model = AutoModel.from_pretrained(model_name)
    model = MultiTaskModel(base_model)

    # Find the latest model file
    model_files = [f for f in os.listdir(save_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {save_dir}")

    model_path = os.path.join(save_dir, sorted(model_files)[-1])
    print(f"Loading weights from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device), tokenizer, device


if __name__ == "__main__":
    # Test model creation
    tokenizer, base_model, device = load_model()
    model = create_multitask_model(base_model, device)

    # Test forward pass
    sample = tokenizer(
        "I feel overwhelmed and alone.",
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(sample['input_ids'], sample['attention_mask'])

    print("\nModel outputs:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tensor.item():.4f}")
