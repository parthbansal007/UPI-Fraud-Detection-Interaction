from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import create_advanced_features
from .preprocess import basic_cleaning, clean_column_names
from .train import OHE_COLS, TARGET_COLS

DEFAULT_MODEL_DIR = Path("models/transaction")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dataframe(df_or_inputs: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    if isinstance(df_or_inputs, pd.DataFrame):
        return df_or_inputs.copy()
    if isinstance(df_or_inputs, dict):
        return pd.DataFrame([df_or_inputs])
    if isinstance(df_or_inputs, list):
        return pd.DataFrame([dict(row) for row in df_or_inputs])
    raise TypeError("df_or_inputs must be a DataFrame, dict, or list[dict].")


def _build_features_for_inference(df: pd.DataFrame, preprocess_artifact: dict[str, Any]) -> pd.DataFrame:
    work = clean_column_names(df)
    work = basic_cleaning(work)

    if "timestamp" not in work.columns:
        work["timestamp"] = "2024-01-01 00:00:00"
    if "hour_of_day" not in work.columns:
        hour = pd.to_datetime(work["timestamp"], errors="coerce").dt.hour.fillna(0)
        work["hour_of_day"] = hour.astype(int)
    if "day_of_week" not in work.columns:
        dow = pd.to_datetime(work["timestamp"], errors="coerce").dt.dayofweek.fillna(0)
        work["day_of_week"] = dow.astype(int)
    if "is_weekend" not in work.columns:
        work["is_weekend"] = (work["day_of_week"].astype(int) >= 5).astype(int)

    required_fallbacks = {
        "transaction_type": "P2P",
        "merchant_category": "general",
        "amount_inr": 0.0,
        "transaction_status": "SUCCESS",
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "unknown",
        "sender_bank": "unknown",
        "receiver_bank": "unknown",
        "device_type": "Mobile",
        "network_type": "4G",
    }
    for col, fallback in required_fallbacks.items():
        if col not in work.columns:
            work[col] = fallback

    work["amount_inr"] = pd.to_numeric(work["amount_inr"], errors="coerce").fillna(0.0)
    engineered = create_advanced_features(work)

    global_mean = float(preprocess_artifact.get("global_mean", 0.0))
    target_maps: dict[str, dict[str, float]] = preprocess_artifact.get("target_maps", {})

    for col in TARGET_COLS:
        if col not in engineered.columns:
            continue
        mapping = target_maps.get(col, {})
        engineered[f"{col}_risk"] = engineered[col].map(mapping).fillna(global_mean)

    existing_ohe = [col for col in OHE_COLS if col in engineered.columns]
    encoded = pd.get_dummies(engineered, columns=existing_ohe, drop_first=True)
    encoded = encoded.select_dtypes(exclude=["object"])
    return encoded


def predict_transaction_risk(
    df_or_inputs: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Transaction model not found: {model_path}")

    scaler_path = model_dir_path / "scaler.pkl"
    iso_path = model_dir_path / "iso_forest.pkl"
    features_path = model_dir_path / "feature_names.pkl"
    preprocess_artifact_path = model_dir_path / "transaction_preprocessing_artifacts.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    iso = joblib.load(iso_path)
    feature_names: list[str] = joblib.load(features_path)
    preprocess_artifact = joblib.load(preprocess_artifact_path) if preprocess_artifact_path.exists() else {}

    input_df = _ensure_dataframe(df_or_inputs)
    built_df = _build_features_for_inference(input_df, preprocess_artifact=preprocess_artifact)

    for feature in feature_names:
        if feature not in built_df.columns:
            built_df[feature] = 0.0

    x_base = scaler.transform(built_df[feature_names])
    iso_scores = (-iso.decision_function(x_base)).reshape(-1, 1)
    x_model = np.hstack([x_base, iso_scores])

    fraud_probability = model.predict_proba(x_model)[:, 1].astype(float)

    threshold_path = model_dir_path / "transaction_thresholds.json"
    threshold_payload = _load_json(threshold_path)
    threshold = float(threshold_payload.get(model_name, 0.5))

    out = pd.DataFrame(
        {
            "session_id": input_df.get("transaction_id", pd.Series([f"tx_{idx:06d}" for idx in range(len(input_df))])).astype(str),
            "transaction_score": fraud_probability,
            "transaction_label": np.where(fraud_probability >= threshold, "fraud", "normal"),
            "transaction_risk_level": np.where(
                fraud_probability >= 0.75,
                "HIGH",
                np.where(fraud_probability >= 0.45, "MEDIUM", "LOW"),
            ),
        }
    )

    if "fraud_flag" in input_df.columns:
        out["true_label"] = pd.to_numeric(input_df["fraud_flag"], errors="coerce").fillna(0).astype(int)
    return out
