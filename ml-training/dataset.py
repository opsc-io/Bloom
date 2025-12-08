"""
Mental Health Dataset Module
Based on phoenix1803/Mental-Health-LongParas dataset
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Tuple, Optional


# Target columns from the dataset
TARGET_COLUMNS = [
    'sentiment_intensity',
    'family_history',
    'trauma_indicators',
    'social_isolation_score',
    'support_system_strength'
]

# Mental health labels for classification
MENTAL_HEALTH_LABELS = [
    "Anxiety",
    "Depression",
    "Suicidal",
    "Stress",
    "Bipolar",
    "Personality disorder",
    "Normal",
]


class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for mental health text classification.

    Targets:
    - sentiment: Regression (-1 to 1)
    - family: Binary classification (0 or 1)
    - trauma: Regression (0 to 7)
    - isolation: Regression (0 to 4)
    - support: Regression (scaled by 100 for training)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row['text'])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Prepare targets
        # Note: We scale 'support' by 100 so the model doesn't predict tiny values
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # Targets:
            'sentiment': torch.tensor(row['sentiment_intensity'], dtype=torch.float),
            'family': torch.tensor(row['family_history'], dtype=torch.float),
            'trauma': torch.tensor(row['trauma_indicators'], dtype=torch.float),
            'isolation': torch.tensor(row['social_isolation_score'], dtype=torch.float),
            'support': torch.tensor(row['support_system_strength'] * 100.0, dtype=torch.float)
        }


def load_mental_health_data(
    dataset_name: str = "phoenix1803/Mental-Health-LongParas",
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split the mental health dataset into train/val/test.

    Args:
        dataset_name: HuggingFace dataset name
        test_size: Fraction for test+val split (split equally)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    df_full = dataset['train'].to_pandas()
    print(f"Full data loaded. Shape: {df_full.shape}")

    # Analyze target columns
    print("\n--- Target Variable Analysis ---")
    for col in TARGET_COLUMNS:
        if col in df_full.columns:
            print(f"{col}: Range [{df_full[col].min():.4f}, {df_full[col].max():.4f}]")

    # Stratified split by label
    v_counts = df_full['label'].value_counts()
    valid_labels = v_counts[v_counts >= 2].index
    df_filtered = df_full[df_full['label'].isin(valid_labels)].copy()

    # 70/30 split first, then split the 30% into val/test (15%/15%)
    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=df_filtered['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df['label']
    )

    print(f"\nTraining Set:   {len(train_df)} rows")
    print(f"Validation Set: {len(val_df)} rows")
    print(f"Testing Set:    {len(test_df)} rows")

    return train_df, val_df, test_df


def load_inference_datasets() -> pd.DataFrame:
    """
    Load additional datasets for inference/labeling.
    Combines counseling conversations and Reddit posts.

    Returns:
        Combined DataFrame with 'processed_text' and 'source' columns
    """
    print("Loading inference datasets...")

    # Dataset A: Counseling conversations
    ds1 = load_dataset("Amod/mental_health_counseling_conversations")
    df1 = ds1['train'].to_pandas()
    df1['processed_text'] = df1['Context']
    df1['source'] = 'counseling_chat'

    # Dataset B: Reddit posts
    ds2 = load_dataset("solomonk/reddit_mental_health_posts")
    df2 = ds2['train'].to_pandas()

    # Combine title and body
    df2['processed_text'] = df2['title'].fillna('') + " " + df2['body'].fillna('')
    df2['source'] = 'reddit_post'

    # Clean Reddit data
    print(f"Original Reddit rows: {len(df2)}")
    df2 = df2[~df2['processed_text'].str.contains(
        r'\[removed\]|\[deleted\]',
        case=False,
        regex=True
    )]
    df2 = df2[df2['processed_text'].str.len() > 50]
    print(f"Valid Reddit rows after cleaning: {len(df2)}")

    # Combine
    df_combined = pd.concat([
        df1[['processed_text', 'source']],
        df2[['processed_text', 'source']]
    ], ignore_index=True)

    print(f"Combined dataset size: {len(df_combined)} rows")
    return df_combined


if __name__ == "__main__":
    # Test dataset loading
    train_df, test_df = load_mental_health_data()
    print("\nSample row:")
    print(train_df.iloc[0][['text', 'label'] + TARGET_COLUMNS])
