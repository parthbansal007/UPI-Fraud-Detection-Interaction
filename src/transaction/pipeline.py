from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .infer import predict_transaction_risk
from .preprocess import preprocess_pipeline
from .shap_explain import run_shap
from .train import SMOOTH_FACTOR, TARGET_COLS, FraudDetector


@dataclass
class TransactionTrainConfig:
    data_path: Path = Path("transaction_data/upi_transactions_2024.csv")
    model_dir: Path = Path("models/transaction")
    docs_dir: Path = Path("docs/transaction")
    output_dir: Path = Path("outputs/transaction")
    model_for_shap: str = "xgboost"
    with_shap: bool = True


@dataclass
class TransactionEvaluateConfig:
    data_path: Path = Path("transaction_data/upi_transactions_2024.csv")
    model_dir: Path = Path("models/transaction")
    output_dir: Path = Path("outputs/transaction")
    model_name: str = "xgboost"


def _fit_target_maps(data_path: Path) -> dict[str, Any]:
    x_df, y = preprocess_pipeline(str(data_path))
    global_mean = float(y.mean())
    target_maps: dict[str, dict[str, float]] = {}

    for col in TARGET_COLS:
        if col not in x_df.columns:
            continue
        agg = y.groupby(x_df[col]).agg(["count", "mean"])
        smoothed = ((agg["count"] * agg["mean"] + SMOOTH_FACTOR * global_mean) / (agg["count"] + SMOOTH_FACTOR)).to_dict()
        target_maps[col] = {str(key): float(value) for key, value in smoothed.items()}

    return {
        "global_mean": global_mean,
        "target_maps": target_maps,
    }


def train_transaction_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = TransactionTrainConfig(**(config or {}))
    cfg.data_path = Path(cfg.data_path)
    cfg.model_dir = Path(cfg.model_dir)
    cfg.docs_dir = Path(cfg.docs_dir)
    cfg.output_dir = Path(cfg.output_dir)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    detector = FraudDetector(model_dir=str(cfg.model_dir), docs_dir=str(cfg.docs_dir))
    results = detector.fit_all(str(cfg.data_path))

    thresholds_path = cfg.model_dir / "transaction_thresholds.json"
    thresholds_path.write_text(json.dumps(detector.thresholds, indent=2), encoding="utf-8")

    preprocess_artifact = _fit_target_maps(cfg.data_path)
    preprocess_path = cfg.model_dir / "transaction_preprocessing_artifacts.joblib"
    joblib.dump(preprocess_artifact, preprocess_path)

    if cfg.with_shap:
        run_shap(
            str(cfg.data_path),
            model_name=cfg.model_for_shap,
            model_dir=str(cfg.model_dir),
            docs_dir=str(cfg.docs_dir),
        )

    report = {
        "status": "trained",
        "data_path": str(cfg.data_path.resolve()),
        "model_dir": str(cfg.model_dir.resolve()),
        "docs_dir": str(cfg.docs_dir.resolve()),
        "thresholds": detector.thresholds,
        "results": {
            model: {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "f2": float(metrics["f2"]),
                "pr_auc": float(metrics["pr_auc"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
            for model, metrics in results.items()
        },
        "artifacts": {
            "thresholds_path": str(thresholds_path.resolve()),
            "preprocessing_path": str(preprocess_path.resolve()),
            "training_report_path": str((cfg.model_dir / "transaction_training_report.json").resolve()),
        },
    }

    training_report_path = cfg.model_dir / "transaction_training_report.json"
    training_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_transaction_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = TransactionEvaluateConfig(**(config or {}))
    cfg.data_path = Path(cfg.data_path)
    cfg.model_dir = Path(cfg.model_dir)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(cfg.data_path)
    pred_df = predict_transaction_risk(raw_df, model_dir=cfg.model_dir, model_name=cfg.model_name)

    if "true_label" not in pred_df.columns:
        raise ValueError("Transaction evaluation requires a fraud_flag/true_label column in the dataset.")

    y_true = pred_df["true_label"].astype(int).to_numpy()
    y_score = pred_df["transaction_score"].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "rows": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    predictions_path = cfg.output_dir / "predictions.csv"
    metrics_path = cfg.output_dir / "metrics.json"
    pred_df.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "status": "evaluated",
        "predictions_path": str(predictions_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
    }
