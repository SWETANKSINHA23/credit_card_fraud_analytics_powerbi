"""
Data Loader and Preprocessing Module for Credit Card Fraud Detection.

This module handles:
- Raw transaction data ingestion from CSV.
- Feature engineering (hour of day extraction, log amount transformation).
- Synthetic business metadata simulation (merchant categories for dashboard analytics).
- Stratified train/test splitting preserving extreme class imbalance.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_COLS: List[str] = [f"V{i}" for i in range(1, 29)] + [
    "log_amount",
    "hour_of_day",
]

MERCHANT_CATEGORIES: List[str] = [
    "E-commerce",
    "Grocery",
    "Travel",
    "Entertainment",
    "Utilities",
    "Restaurant",
]
FRAUD_CATEGORY_WEIGHTS: List[float] = [0.35, 0.05, 0.30, 0.15, 0.05, 0.10]
LEGIT_CATEGORY_WEIGHTS: List[float] = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]


def get_feature_columns() -> List[str]:
    """Return the list of feature column names used for model training and inference."""
    return list(DEFAULT_FEATURE_COLS)


def load_raw_data(data_path: Union[str, Path] = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load raw credit card transaction data from a CSV file.

    Args:
        data_path: Filepath to creditcard.csv.

    Returns:
        pd.DataFrame containing raw transaction data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'. "
            "Please ensure creditcard.csv is placed in the data/ directory."
        )

    df = pd.read_csv(path)
    return df


def engineer_features(
    df: pd.DataFrame,
    add_merchant_category: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply feature engineering transformations to the raw dataset.

    Transformations:
    - hour_of_day: Cyclical intraday time feature derived from Time (in seconds).
    - log_amount: Log1p-transformed Amount feature to compress extreme transaction skew.
    - merchant_category: (Optional) Simulated merchant category for Power BI drill-down.

    Args:
        df: Input DataFrame containing 'Time', 'Amount', and optionally 'Class'.
        add_merchant_category: Whether to simulate merchant categories.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with engineered features added.
    """
    df_engineered = df.copy()

    # Intraday hour calculation from elapsed seconds
    if "Time" in df_engineered.columns:
        df_engineered["hour_of_day"] = (df_engineered["Time"] // 3600) % 24
    elif "hour_of_day" not in df_engineered.columns:
        df_engineered["hour_of_day"] = 0.0

    # Log1p transformation for transaction amount
    if "Amount" in df_engineered.columns:
        df_engineered["log_amount"] = np.log1p(df_engineered["Amount"])
    elif "log_amount" not in df_engineered.columns:
        df_engineered["log_amount"] = 0.0

    # Simulate merchant categories for dashboard analytics if 'Class' is present
    if add_merchant_category and "Class" in df_engineered.columns:
        rng = np.random.default_rng(random_state)
        n_rows = len(df_engineered)
        fraud_mask = df_engineered["Class"] == 1

        fraud_cats = rng.choice(
            MERCHANT_CATEGORIES, size=n_rows, p=FRAUD_CATEGORY_WEIGHTS
        )
        legit_cats = rng.choice(
            MERCHANT_CATEGORIES, size=n_rows, p=LEGIT_CATEGORY_WEIGHTS
        )

        df_engineered["merchant_category"] = np.where(
            fraud_mask, fraud_cats, legit_cats
        )

    return df_engineered


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Class",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into stratified train and test partitions.

    Args:
        df: Input DataFrame with engineered features and target label.
        test_size: Proportion of dataset to include in test split (default: 0.2).
        random_state: Random seed for reproducibility.
        feature_cols: List of feature column names. If None, uses DEFAULT_FEATURE_COLS.
        target_col: Target column name (default: 'Class').

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def load_and_preprocess_pipeline(
    data_path: Union[str, Path] = "data/creditcard.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience end-to-end data loading and preprocessing pipeline.

    Returns:
        Tuple of (full_engineered_df, X_train, X_test, y_train, y_test).
    """
    raw_df = load_raw_data(data_path)
    engineered_df = engineer_features(
        raw_df, add_merchant_category=True, random_state=random_state
    )
    X_train, X_test, y_train, y_test = split_data(
        engineered_df,
        test_size=test_size,
        random_state=random_state,
    )
    return engineered_df, X_train, X_test, y_train, y_test
