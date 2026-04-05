from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

LABEL_ENCODING = {"normal": 0, "suspicious": 1, "malicious": 2}
LEAKAGE_COLUMNS = ["risk_score", "split", "split_reason", "generator_version"]

REQUIRED_RAW_COLUMNS = [
    "session_id",
    "event_sequence",
    "event_timestamps",
    "device_state",
    "behavioral_features",
    "label",
    "scenario_family",
    "risk_score",
    "split",
]

REQUIRED_ENCODED_COLUMNS = [
    "session_id",
    "vpn",
    "rooted",
    "emulator",
    "ip_risk_score",
    "session_duration",
    "num_events",
    "num_links_clicked",
    "num_qr_scans",
    "num_permission_requests",
]


def _resolve_labels(encoded_df: pd.DataFrame) -> pd.Series:
    if "label" in encoded_df.columns:
        labels = encoded_df["label"].astype(str).str.strip().str.lower()
        encoded = labels.map(LABEL_ENCODING)
        if encoded.isna().any():
            unknown_labels = sorted(labels[encoded.isna()].unique().tolist())
            raise ValueError(f"Unknown labels found in encoded dataset: {unknown_labels}")
        return encoded.astype("int64")

    if "label_id" in encoded_df.columns:
        label_id = pd.to_numeric(encoded_df["label_id"], errors="coerce")
        if label_id.isna().any():
            raise ValueError("label_id contains null or non-numeric values.")
        allowed = set(LABEL_ENCODING.values())
        observed = set(label_id.astype("int64").unique().tolist())
        if not observed.issubset(allowed):
            invalid = sorted(observed - allowed)
            raise ValueError(f"label_id contains unsupported classes: {invalid}")
        return label_id.astype("int64")

    raise ValueError("dataset_encoded_v2.csv must include either 'label' or 'label_id'.")


def _fill_feature_nulls(features: pd.DataFrame) -> pd.DataFrame:
    clean = features.copy()
    numeric_cols = clean.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in clean.columns if col not in numeric_cols]
    if numeric_cols:
        clean.loc[:, numeric_cols] = clean[numeric_cols].fillna(0)
    if categorical_cols:
        clean.loc[:, categorical_cols] = clean[categorical_cols].fillna("unknown")
    return clean


def prepare_training_datasets(
    data_dir: str | Path = "interaction_data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    paths = resolve_interaction_paths(data_dir=data_dir)
    encoded_path = paths["encoded_data_path"]
    if not encoded_path.exists():
        raise FileNotFoundError(f"Missing interaction encoded dataset: {encoded_path}")

    encoded_df = pd.read_csv(encoded_path)
    if "split" not in encoded_df.columns:
        raise ValueError("dataset_encoded_v2.csv is missing required column: split")

    labels = _resolve_labels(encoded_df)
    split_values = set(encoded_df["split"].dropna().astype(str).unique().tolist())
    required_splits = {"train", "val", "test_ood"}
    missing_splits = sorted(required_splits - split_values)
    if missing_splits:
        raise ValueError(f"dataset_encoded_v2.csv missing required split values: {missing_splits}")

    drop_columns = set(LEAKAGE_COLUMNS + ["label", "label_id"])
    feature_columns = [col for col in encoded_df.columns if col not in drop_columns]
    features = _fill_feature_nulls(encoded_df[feature_columns])

    split_series = encoded_df["split"].astype(str).str.strip()
    train_mask = split_series == "train"
    val_mask = split_series == "val"
    test_mask = split_series == "test_ood"

    X_train = features.loc[train_mask].reset_index(drop=True)
    X_val = features.loc[val_mask].reset_index(drop=True)
    X_test = features.loc[test_mask].reset_index(drop=True)
    y_train = labels.loc[train_mask].reset_index(drop=True)
    y_val = labels.loc[val_mask].reset_index(drop=True)
    y_test = labels.loc[test_mask].reset_index(drop=True)

    datasets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    null_violations = {name: int(frame_or_series.isna().sum().sum()) for name, frame_or_series in datasets.items()}
    invalid = {name: count for name, count in null_violations.items() if count > 0}
    if invalid:
        raise ValueError(f"Null values remain after preprocessing: {invalid}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def resolve_interaction_paths(data_dir: str | Path = "interaction_data") -> dict[str, Path]:
    root = Path(data_dir)
    return {
        "data_dir": root,
        "raw_data_path": root / "dataset_v2.csv",
        "encoded_data_path": root / "dataset_encoded_v2.csv",
    }


def load_interaction_datasets(data_dir: str | Path = "interaction_data") -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_interaction_paths(data_dir)
    raw_path = paths["raw_data_path"]
    encoded_path = paths["encoded_data_path"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing interaction raw dataset: {raw_path}")
    if not encoded_path.exists():
        raise FileNotFoundError(f"Missing interaction encoded dataset: {encoded_path}")

    raw_df = pd.read_csv(raw_path)
    encoded_df = pd.read_csv(encoded_path)
    return raw_df, encoded_df


def validate_interaction_datasets(raw_df: pd.DataFrame, encoded_df: pd.DataFrame) -> dict[str, Any]:
    raw_missing = [col for col in REQUIRED_RAW_COLUMNS if col not in raw_df.columns]
    encoded_missing = [col for col in REQUIRED_ENCODED_COLUMNS if col not in encoded_df.columns]

    if raw_missing:
        raise ValueError(f"dataset_v2.csv missing required columns: {raw_missing}")
    if encoded_missing:
        raise ValueError(f"dataset_encoded_v2.csv missing required columns: {encoded_missing}")

    split_values = set(raw_df["split"].dropna().astype(str).unique().tolist())
    required_splits = {"train", "val", "test_ood"}
    missing_splits = sorted(required_splits - split_values)
    if missing_splits:
        raise ValueError(f"dataset_v2.csv missing required split values: {missing_splits}")

    return {
        "raw_rows": int(len(raw_df)),
        "encoded_rows": int(len(encoded_df)),
        "raw_columns": list(raw_df.columns),
        "encoded_columns": list(encoded_df.columns),
        "label_distribution": raw_df["label"].value_counts(dropna=False).to_dict(),
        "split_distribution": raw_df["split"].value_counts(dropna=False).to_dict(),
    }


def prepare_interaction_data(data_dir: str | Path = "interaction_data") -> dict[str, Any]:
    raw_df, encoded_df = load_interaction_datasets(data_dir=data_dir)
    validation = validate_interaction_datasets(raw_df=raw_df, encoded_df=encoded_df)
    return {
        "raw_df": raw_df,
        "encoded_df": encoded_df,
        "validation": validation,
        "paths": resolve_interaction_paths(data_dir),
    }
