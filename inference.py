from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from feature_pipeline import suspicious_subsequence_score
from model_training import explain_prediction


def load_artifacts(
    model_path: str = "trained_model.pkl",
    feature_columns_path: str = "feature_columns.pkl",
    shap_explainer_path: str = "shap_explainer.pkl",
) -> tuple[Any, list[str], Any]:
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)
    shap_explainer = joblib.load(shap_explainer_path)
    return model, feature_columns, shap_explainer


def parse_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = ast.literal_eval(text)
    return parsed if isinstance(parsed, list) else []


def derive_payload_features(payload: dict[str, Any]) -> dict[str, float]:
    enriched = dict(payload)
    num_events = float(enriched.get("num_events", 0.0) or 0.0)
    num_permission_requests = float(enriched.get("num_permission_requests", 0.0) or 0.0)
    num_links_clicked = float(enriched.get("num_links_clicked", 0.0) or 0.0)
    num_qr_scans = float(enriched.get("num_qr_scans", 0.0) or 0.0)
    session_duration = float(enriched.get("session_duration", 0.0) or 0.0)
    encoded_events = parse_sequence(enriched.get("event_sequence_encoded"))
    raw_events = parse_sequence(enriched.get("event_sequence"))
    enriched["permission_abuse_ratio"] = num_permission_requests / num_events if num_events else 0.0
    enriched["link_click_rate"] = num_links_clicked / num_events if num_events else 0.0
    enriched["qr_to_action_ratio"] = num_qr_scans / num_events if num_events else 0.0
    enriched["rapid_activity_flag"] = float((session_duration / num_events) < 5.0) if num_events else 0.0
    if raw_events:
        enriched["event_diversity"] = float(len(set(str(item) for item in raw_events)))
        enriched["suspicious_pattern_score"] = float(suspicious_subsequence_score([str(item) for item in raw_events]))
    elif encoded_events:
        enriched["event_diversity"] = float(len(set(encoded_events)))
        enriched["suspicious_pattern_score"] = float(enriched.get("suspicious_pattern_score", 0.0) or 0.0)
    else:
        enriched["event_diversity"] = float(enriched.get("event_diversity", 0.0) or 0.0)
        enriched["suspicious_pattern_score"] = float(enriched.get("suspicious_pattern_score", 0.0) or 0.0)
    return enriched


def build_feature_frame(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    enriched = derive_payload_features(payload)
    row = {}
    for column in feature_columns:
        row[column] = float(enriched.get(column, 0.0) or 0.0)
    return pd.DataFrame([row], columns=feature_columns)


def predict_from_payload(
    payload: dict[str, Any],
    model_path: str = "trained_model.pkl",
    feature_columns_path: str = "feature_columns.pkl",
    shap_explainer_path: str = "shap_explainer.pkl",
) -> dict[str, Any]:
    model, feature_columns, shap_explainer = load_artifacts(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        shap_explainer_path=shap_explainer_path,
    )
    frame = build_feature_frame(payload, feature_columns)
    scored = model.score_frame(frame).iloc[0]
    explanation = explain_prediction(frame.iloc[0], model, shap_explainer)
    return {
        "predicted_label": str(scored["predicted_label"]),
        "risk_level": str(scored["risk_level"]),
        "model_score": float(scored["model_score"]),
        "rule_score": float(scored["rule_score"]),
        "final_score": float(scored["final_score"]),
        "probabilities": {
            "normal": float(scored["prob_normal"]),
            "suspicious": float(scored["prob_suspicious"]),
            "malicious": float(scored["prob_malicious"]),
        },
        "explanation": explanation,
    }


def predict_from_json(
    payload_path: str,
    model_path: str = "trained_model.pkl",
    feature_columns_path: str = "feature_columns.pkl",
    shap_explainer_path: str = "shap_explainer.pkl",
) -> dict[str, Any]:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    return predict_from_payload(
        payload,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        shap_explainer_path=shap_explainer_path,
    )
