from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .fusion import infer_unified_risk

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class UnifiedEvaluateConfig:
    input_path: Path = Path("transaction_data/upi_transactions_2024.csv")
    output_dir: Path = Path("outputs/unified")
    positive_label_threshold: float = 0.50


def _binary_targets(df: pd.DataFrame) -> np.ndarray:
    if "fraud_flag" in df.columns:
        return pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0).astype(int).to_numpy()

    if "true_label" in df.columns:
        raw = df["true_label"].astype(str).str.lower()
        return raw.isin(["1", "fraud", "malicious", "suspicious", "high"]).astype(int).to_numpy()

    raise ValueError("Unified evaluation requires fraud_flag or true_label.")


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], ["normal", "fraud"])
    ax.set_yticks([0, 1], ["normal", "fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Unified Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Unified ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pr(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(recall, precision, label=f"PR AUC={pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Unified Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_calibration(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="fused")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Unified Calibration")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_unified_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = UnifiedEvaluateConfig(**(config or {}))
    cfg.input_path = Path(cfg.input_path)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(cfg.input_path)
    pred_df = infer_unified_risk(raw_df)

    y_true = _binary_targets(raw_df)
    y_score = pd.to_numeric(pred_df["fused_risk_score"], errors="coerce").fillna(0.0).to_numpy()
    y_pred = (y_score >= float(cfg.positive_label_threshold)).astype(int)

    interaction_score = pd.to_numeric(pred_df["interaction_score"], errors="coerce").fillna(0.0).to_numpy()
    transaction_score = pd.to_numeric(pred_df["transaction_score"], errors="coerce").fillna(0.0).to_numpy()

    def _metric_payload(score: np.ndarray) -> dict[str, float]:
        pred = (score >= float(cfg.positive_label_threshold)).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, score)),
            "roc_auc": float(roc_auc_score(y_true, score)),
        }

    metrics = {
        "rows": int(len(pred_df)),
        "positive_label_threshold": float(cfg.positive_label_threshold),
        "interaction": _metric_payload(interaction_score),
        "transaction": _metric_payload(transaction_score),
        "fused": _metric_payload(y_score),
        "fused_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    predictions_path = cfg.output_dir / "predictions.csv"
    metrics_path = cfg.output_dir / "metrics.json"
    eval_graph_dir = cfg.output_dir / "evaluation_graphs"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df["fraud_flag"] = y_true
    pred_df.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _save_confusion(y_true, y_pred, eval_graph_dir / "confusion_matrix.png")
    _save_roc(y_true, y_score, eval_graph_dir / "roc_curve.png")
    _save_pr(y_true, y_score, eval_graph_dir / "precision_recall_curve.png")
    _save_calibration(y_true, y_score, eval_graph_dir / "calibration_curve.png")

    return {
        "status": "evaluated",
        "predictions_path": str(predictions_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "evaluation_graphs_dir": str(eval_graph_dir.resolve()),
        "metrics": metrics,
    }
