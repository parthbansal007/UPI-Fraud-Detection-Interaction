"""
Loads and cleans the UPI transactions CSV.
Encoding is deferred to train.py to prevent target-leakage.
"""

import pandas as pd
from src.transaction.feature_engineering import create_advanced_features


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\(\)]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["transaction_id"], errors="ignore")
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    # Rename columns to canonical names used throughout the codebase
    rename_map = {
        "amount_inr_": "amount_inr",
        "transaction_type": "transaction_type",
    }
    # Handle "amount (INR)" → "amount_inr" after clean_column_names
    if "amount_inr_" in df.columns:
        df = df.rename(columns={"amount_inr_": "amount_inr"})
    return df


def split_features_target(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=["timestamp"], errors="ignore")
    X = df.drop(columns=["fraud_flag"])
    y = df["fraud_flag"]
    return X, y


def preprocess_pipeline(path: str):
    """
    Full pipeline:
      load → clean columns → basic cleaning → feature engineering → X/y split
    """
    df = load_data(path)
    df = clean_column_names(df)
    df = basic_cleaning(df)
    df = create_advanced_features(df)
    X, y = split_features_target(df)
    return X, y