from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .preprocess import resolve_interaction_paths
from .train_model import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, load_interaction_model

LABEL_ORDER = ["normal", "suspicious", "malicious"]


def build_feature_frame_for_split(
    model: Any,
    data_dir: str | Path = "interaction_data",
    split: str = "test_ood",
) -> pd.DataFrame:
    paths = resolve_interaction_paths(data_dir=data_dir)
    prev_raw = model.config.raw_data_path
    prev_encoded = model.config.encoded_data_path
    try:
        model.config.raw_data_path = paths["raw_data_path"]
        model.config.encoded_data_path = paths["encoded_data_path"]
        merged = model._load_data()
        feature_df = model._build_feature_table(merged)
    finally:
        model.config.raw_data_path = prev_raw
        model.config.encoded_data_path = prev_encoded

    split_df = feature_df[feature_df["split"] == split].reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"No rows found for split '{split}'.")
    return split_df


def _build_metrics_payload(
    y_true: pd.Series,
    y_pred: pd.Series,
    split: str,
    y_score: pd.Series | None = None,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    try:
        y_bin = (y_true == "malicious").astype(int)
        score = y_score.astype(float) if y_score is not None else (y_pred == "malicious").astype(int)
        auc = roc_auc_score(y_bin, score)
    except:
        auc = 0.0
    
    return {
        "split": split,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
        "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
        "malicious_precision": float(report["malicious"]["precision"]),
        "malicious_recall": float(report["malicious"]["recall"]),"roc_auc": float(auc),
    }


def _build_detailed_metrics_payload(y_true: pd.Series, y_pred: pd.Series, split: str) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist()
    return {
        "split": split,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
        "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
        "malicious_precision": float(report["malicious"]["precision"]),
        "malicious_recall": float(report["malicious"]["recall"]),
        "confusion_matrix_labels": LABEL_ORDER,
        "confusion_matrix": matrix,
    }


def evaluate_interaction_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or {})
    model_dir = Path(cfg.get("model_dir", DEFAULT_MODEL_DIR))
    output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    data_dir = Path(cfg.get("data_dir", "interaction_data"))
    split = str(cfg.get("split", "test_ood"))
    generate_xai = bool(cfg.get("generate_xai", False))

    model = load_interaction_model(model_dir=model_dir)
    split_df = build_feature_frame_for_split(model=model, data_dir=data_dir, split=split)
    pred_df = model.predict_table(split_df)

    standardized = pd.DataFrame(
        {
            "session_id": pred_df["session_id"],
            "interaction_label": pred_df["predicted_label"],
            "interaction_risk_score": pred_df["fraud_probability"],
            "interaction_risk_level": pred_df["risk_level"],
            "normal_probability": pred_df["normal_probability"],
            "suspicious_probability": pred_df["suspicious_probability"],
            "malicious_probability": pred_df["malicious_probability"],
            "true_label": pred_df["label"],
            "source_split": pred_df["split"],
            "scenario_family": pred_df["scenario_family"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    standardized.to_csv(predictions_path, index=False)

    metrics_payload = _build_metrics_payload(
        pred_df["label"],
        pred_df["predicted_label"],
        split=split,
        y_score=pred_df["fraud_probability"],
    )
    metrics_payload["model_dir"] = str(model_dir.resolve())
    metrics_payload["predictions_path"] = str(predictions_path.resolve())

    train_report_path = model_dir / "evaluation_report.json"
    if train_report_path.exists():
        metrics_payload["training_report_path"] = str(train_report_path.resolve())
        metrics_payload["training_summary"] = json.loads(train_report_path.read_text(encoding="utf-8")).get("test_ood_metrics", {})

    if generate_xai:
        from .xai import export_xai_artifacts

        metrics_payload["xai"] = export_xai_artifacts(
            model_dir=model_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            split=split,
            sample_size=int(cfg.get("xai_sample_size", 200)),
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def evaluate_fraud_detection_system(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or {})
    predictions_path = Path(cfg.get("predictions_path", "outputs/ensemble_fraud_probabilities.csv"))
    output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}. Run ensemble scoring first."
        )

    pred_df = pd.read_csv(predictions_path)
    required_columns = {"split", "label", "predicted_label"}
    missing = sorted(required_columns - set(pred_df.columns))
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    payload: dict[str, Any] = {
        "source_predictions": str(predictions_path.resolve()),
        "class_order": LABEL_ORDER,
        "focus": "malicious precision and recall",
    }

    split_map = {"validation": "val", "test_ood": "test_ood"}
    for key, split_name in split_map.items():
        split_df = pred_df[pred_df["split"] == split_name].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No rows found for split '{split_name}' in {predictions_path}.")
        y_true = split_df["label"].astype(str).str.lower()
        y_pred = split_df["predicted_label"].astype(str).str.lower()
        payload[key] = _build_detailed_metrics_payload(y_true=y_true, y_pred=y_pred, split=split_name)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["metrics_path"] = str(metrics_path.resolve())
    return payload
