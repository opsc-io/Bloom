"""
Gradio Demo Application for Mental Health Analysis Model
For project submission and live demonstration.

CRISP-DM Phase: Deployment (Demo Interface)
"""

import os
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import plotly.graph_objects as go
import plotly.express as px

# Model configuration
MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 512

# Sample texts for demo
SAMPLE_TEXTS = [
    "I've been feeling really down lately. It's hard to get out of bed and I don't want to talk to anyone.",
    "Therapy has been helping so much! I finally feel like I'm making progress and my family is supportive.",
    "The anxiety keeps getting worse. I can't sleep and I'm constantly worried about everything.",
    "Had a great session today. We talked about coping strategies and I feel more hopeful.",
    "I feel completely alone. Nobody understands what I'm going through and I've pushed everyone away.",
]


class MultiTaskModel(nn.Module):
    """Multi-task model with XLM-RoBERTa backbone and task-specific heads."""

    def __init__(self, pretrained_model_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.head_sentiment = nn.Linear(self.hidden_size, 1)
        self.head_trauma = nn.Linear(self.hidden_size, 1)
        self.head_isolation = nn.Linear(self.hidden_size, 1)
        self.head_support = nn.Linear(self.hidden_size, 1)
        self.head_family = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        return {
            "sentiment": torch.tanh(self.head_sentiment(pooled)).squeeze(-1),
            "trauma": torch.relu(self.head_trauma(pooled)).squeeze(-1) * 7,
            "isolation": torch.relu(self.head_isolation(pooled)).squeeze(-1) * 4,
            "support": torch.sigmoid(self.head_support(pooled)).squeeze(-1),
            "family": torch.sigmoid(self.head_family(pooled)).squeeze(-1),
        }

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(pretrained_model_name=MODEL_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


# Global model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str = None):
    """Load the model and tokenizer."""
    global model, tokenizer

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = MultiTaskModel.load(model_path, device=device)
    else:
        print("Creating new model (for demo purposes)")
        model = MultiTaskModel()

    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


def analyze_text(text: str) -> tuple:
    """Analyze text and return predictions with visualizations."""
    if model is None or tokenizer is None:
        load_model()

    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    predictions = {
        "Sentiment": float(outputs["sentiment"].cpu().item()),
        "Trauma Level": float(outputs["trauma"].cpu().item()),
        "Social Isolation": float(outputs["isolation"].cpu().item()),
        "Support System": float(outputs["support"].cpu().item()),
        "Family History Prob.": float(outputs["family"].cpu().item()),
    }

    # Determine risk level
    risk_level = compute_risk_level(predictions)

    # Create visualizations
    radar_chart = create_radar_chart(predictions)
    bar_chart = create_bar_chart(predictions)
    risk_gauge = create_risk_gauge(risk_level)

    # Format text output
    results_text = format_results(predictions, risk_level)

    return results_text, radar_chart, bar_chart, risk_gauge


def compute_risk_level(pred: dict) -> str:
    """Compute overall risk level."""
    sentiment = pred["Sentiment"]
    trauma = pred["Trauma Level"]
    isolation = pred["Social Isolation"]
    support = pred["Support System"]

    if sentiment < -0.5 and trauma > 4:
        return "High"
    if isolation > 3 and support < 0.3:
        return "High"
    if trauma > 5:
        return "High"

    if sentiment < 0 and trauma > 2:
        return "Medium"
    if isolation > 2:
        return "Medium"

    return "Low"


def create_radar_chart(predictions: dict) -> go.Figure:
    """Create radar chart of predictions."""
    categories = list(predictions.keys())

    # Normalize values to 0-1 scale for radar chart
    normalized = {
        "Sentiment": (predictions["Sentiment"] + 1) / 2,  # -1 to 1 -> 0 to 1
        "Trauma Level": predictions["Trauma Level"] / 7,   # 0 to 7 -> 0 to 1
        "Social Isolation": predictions["Social Isolation"] / 4,  # 0 to 4 -> 0 to 1
        "Support System": predictions["Support System"],  # already 0 to 1
        "Family History Prob.": predictions["Family History Prob."],  # already 0 to 1
    }

    values = [normalized[cat] for cat in categories]
    values.append(values[0])  # Close the radar
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Mental Health Indicators',
        line_color='#1f77b4'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="Mental Health Indicator Profile"
    )

    return fig


def create_bar_chart(predictions: dict) -> go.Figure:
    """Create bar chart of predictions."""
    categories = list(predictions.keys())
    values = list(predictions.values())

    # Color based on concern level
    colors = []
    for cat, val in predictions.items():
        if cat == "Sentiment":
            colors.append("#e74c3c" if val < -0.3 else "#2ecc71" if val > 0.3 else "#f1c40f")
        elif cat == "Trauma Level":
            colors.append("#e74c3c" if val > 4 else "#2ecc71" if val < 2 else "#f1c40f")
        elif cat == "Social Isolation":
            colors.append("#e74c3c" if val > 2.5 else "#2ecc71" if val < 1 else "#f1c40f")
        elif cat == "Support System":
            colors.append("#2ecc71" if val > 0.6 else "#e74c3c" if val < 0.3 else "#f1c40f")
        else:
            colors.append("#3498db")

    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors)
    ])

    fig.update_layout(
        title="Prediction Values",
        yaxis_title="Score",
        showlegend=False
    )

    return fig


def create_risk_gauge(risk_level: str) -> go.Figure:
    """Create gauge chart for risk level."""
    value_map = {"Low": 25, "Medium": 50, "High": 85}
    color_map = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value_map[risk_level],
        title={'text': f"Risk Level: {risk_level}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color_map[risk_level]},
            'steps': [
                {'range': [0, 33], 'color': "#d4edda"},
                {'range': [33, 66], 'color': "#fff3cd"},
                {'range': [66, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def format_results(predictions: dict, risk_level: str) -> str:
    """Format predictions as readable text."""
    output = "## Analysis Results\n\n"

    output += f"**Overall Risk Level: {risk_level}**\n\n"

    output += "### Detailed Scores:\n\n"
    output += f"- **Sentiment**: {predictions['Sentiment']:.3f} "
    output += "(Negative)" if predictions['Sentiment'] < -0.3 else "(Positive)" if predictions['Sentiment'] > 0.3 else "(Neutral)"
    output += "\n"

    output += f"- **Trauma Indicators**: {predictions['Trauma Level']:.2f}/7 "
    output += "(High)" if predictions['Trauma Level'] > 4 else "(Low)" if predictions['Trauma Level'] < 2 else "(Moderate)"
    output += "\n"

    output += f"- **Social Isolation**: {predictions['Social Isolation']:.2f}/4 "
    output += "(High)" if predictions['Social Isolation'] > 2.5 else "(Low)" if predictions['Social Isolation'] < 1 else "(Moderate)"
    output += "\n"

    output += f"- **Support System**: {predictions['Support System']:.2f} "
    output += "(Strong)" if predictions['Support System'] > 0.6 else "(Weak)" if predictions['Support System'] < 0.3 else "(Moderate)"
    output += "\n"

    output += f"- **Family History Probability**: {predictions['Family History Prob.']:.1%}\n"

    return output


# Create Gradio interface
def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(title="Bloom Mental Health Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Bloom Mental Health Text Analyzer

        This AI model analyzes text for mental health indicators, designed for therapist-patient
        conversation analysis in the Bloom Health platform.

        **Model Architecture:** XLM-RoBERTa Large with Multi-Task Learning

        **Predictions:**
        - Sentiment (-1 to 1)
        - Trauma indicators (0-7)
        - Social isolation (0-4)
        - Support system strength (0-1)
        - Family history probability (0-1)
        """)

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Type or paste text here...",
                    lines=5
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Examples(
                    examples=SAMPLE_TEXTS,
                    inputs=text_input,
                    label="Sample Texts"
                )

            with gr.Column(scale=3):
                results_output = gr.Markdown(label="Results")

        with gr.Row():
            radar_plot = gr.Plot(label="Indicator Profile")
            bar_plot = gr.Plot(label="Score Breakdown")
            gauge_plot = gr.Plot(label="Risk Assessment")

        # Connect buttons
        analyze_btn.click(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[results_output, radar_plot, bar_plot, gauge_plot]
        )

        clear_btn.click(
            fn=lambda: ("", None, None, None),
            outputs=[results_output, radar_plot, bar_plot, gauge_plot]
        )

        gr.Markdown("""
        ---
        ### About This Model

        - **Training Data:** 120,000 labeled mental health conversations
        - **Validation Loss:** 0.2417
        - **Sentiment R²:** 0.72
        - **Trauma R²:** 0.52
        - **Family History F1:** 0.62

        Built for the Bloom Health Platform using CRISP-DM methodology.

        *This is a demonstration model. For clinical use, always consult licensed professionals.*
        """)

    return demo


if __name__ == "__main__":
    # Load model
    model_path = os.getenv("MODEL_PATH", "mental_health_model_full.pt")
    load_model(model_path if os.path.exists(model_path) else None)

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Creates public link for demo
    )
