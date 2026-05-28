from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_ORDER = ["normal", "suspicious", "malicious"]
PROBABILITY_COLUMNS = {
    "normal": "normal_probability",
    "suspicious": "suspicious_probability",
    "malicious": "malicious_probability",
}


@dataclass
class EvaluationGraphConfig:
    predictions_path: Path = Path("outputs/predictions.csv")
    output_dir: Path = Path("outputs/evaluation_graphs")
    metadata_path: Path = Path("models/interaction/metadata.json")
    dpi: int = 180


def _style_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _load_metadata_threshold(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = metadata.get("malicious_threshold")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)
    required = {"true_label", "interaction_label", *PROBABILITY_COLUMNS.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    out = df.copy()
    out["true_label"] = out["true_label"].astype(str).str.strip().str.lower()
    out["interaction_label"] = out["interaction_label"].astype(str).str.strip().str.lower()
    out = out[out["true_label"].isin(LABEL_ORDER) & out["interaction_label"].isin(LABEL_ORDER)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No labeled rows available for evaluation graphs.")

    for col in PROBABILITY_COLUMNS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(PROBABILITY_COLUMNS.values())].isna().any().any():
        raise ValueError("Probability columns contain non-numeric or missing values.")

    return out


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    dpi: int,
    normalize: bool = False,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    values = cm.astype(float)
    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = values.max() * 0.55 if values.size else 0
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            text = f"{values[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            color = "white" if values[i, j] > threshold else "#1f2933"
            ax.text(j, i, text, ha="center", va="center", color=color, fontweight="semibold")

    _save(fig, path, dpi)


def _plot_class_metric_bars(report: dict[str, Any], path: Path, dpi: int) -> None:
    metrics = ["precision", "recall", "f1-score"]
    data = np.array([[float(report[label][metric]) for metric in metrics] for label in LABEL_ORDER])
    x = np.arange(len(LABEL_ORDER))
    width = 0.24
    colors = ["#2f6fed", "#17a673", "#f59f00"]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, data[:, idx], width=width, label=metric, color=colors[idx])

    ax.set_title("Per-Class Precision, Recall, and F1")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x, LABEL_ORDER)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    _save(fig, path, dpi)


def _plot_roc_curves(y_true: np.ndarray, probabilities: pd.DataFrame, path: Path, dpi: int) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.6, 6.1))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7b8794", linewidth=1.2, label="random")
    auc_by_class: dict[str, float] = {}

    for label in LABEL_ORDER:
        y_bin = (y_true == label).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        score = probabilities[PROBABILITY_COLUMNS[label]].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_bin, score)
        roc_auc = float(auc(fpr, tpr))
        auc_by_class[label] = roc_auc
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} AUC={roc_auc:.3f}")

    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _save(fig, path, dpi)
    return auc_by_class


def _plot_precision_recall_curves(
    y_true: np.ndarray,
    probabilities: pd.DataFrame,
    path: Path,
    dpi: int,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.6, 6.1))
    ap_by_class: dict[str, float] = {}

    for label in LABEL_ORDER:
        y_bin = (y_true == label).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        score = probabilities[PROBABILITY_COLUMNS[label]].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_bin, score)
        avg_precision = float(average_precision_score(y_bin, score))
        ap_by_class[label] = avg_precision
        ax.plot(recall, precision, linewidth=2, label=f"{label} AP={avg_precision:.3f}")

    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    _save(fig, path, dpi)
    return ap_by_class


def _plot_malicious_threshold_tradeoff(
    y_true: np.ndarray,
    malicious_score: np.ndarray,
    path: Path,
    dpi: int,
    selected_threshold: float | None,
) -> dict[str, float]:
    y_bin = (y_true == "malicious").astype(int)
    thresholds = np.linspace(0.0, 1.0, 201)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    accuracy_values: list[float] = []

    for threshold in thresholds:
        pred = malicious_score >= threshold
        tp = np.logical_and(y_bin == 1, pred).sum()
        fp = np.logical_and(y_bin == 0, pred).sum()
        fn = np.logical_and(y_bin == 1, ~pred).sum()
        tn = np.logical_and(y_bin == 0, ~pred).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / max(len(y_bin), 1)
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        accuracy_values.append(float(accuracy))

    best_idx = int(np.argmax(f1_values))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(thresholds, precision_values, label="precision", linewidth=2)
    ax.plot(thresholds, recall_values, label="recall", linewidth=2)
    ax.plot(thresholds, f1_values, label="F1", linewidth=2)
    ax.plot(thresholds, accuracy_values, label="binary accuracy", linewidth=1.8, linestyle="--")
    ax.axvline(thresholds[best_idx], color="#111827", linestyle=":", linewidth=1.8, label=f"best F1={thresholds[best_idx]:.2f}")
    if selected_threshold is not None:
        ax.axvline(selected_threshold, color="#d9480f", linestyle="-.", linewidth=1.8, label=f"selected={selected_threshold:.2f}")
    ax.set_title("Malicious Threshold Tradeoff")
    ax.set_xlabel("Malicious probability threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    _save(fig, path, dpi)

    return {
        "best_f1_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_values[best_idx]),
        "selected_threshold": selected_threshold,
    }


def _plot_score_distribution(df: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = np.linspace(0, 1, 31)
    colors = {"normal": "#2f6fed", "suspicious": "#f59f00", "malicious": "#d6336c"}
    for label in LABEL_ORDER:
        values = df.loc[df["true_label"] == label, "malicious_probability"].to_numpy(dtype=float)
        ax.hist(values, bins=bins, alpha=0.42, density=True, label=label, color=colors[label])
    ax.set_title("Malicious Score Distribution by True Label")
    ax.set_xlabel("Malicious probability")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    _save(fig, path, dpi)


def _plot_malicious_calibration(y_true: np.ndarray, malicious_score: np.ndarray, path: Path, dpi: int) -> None:
    y_bin = (y_true == "malicious").astype(int)
    prob_true, prob_pred = calibration_curve(y_bin, malicious_score, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7b8794", linewidth=1.2, label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#2f6fed", label="model")
    ax.set_title("Malicious Probability Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed malicious rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    _save(fig, path, dpi)


def _plot_confidence_distribution(df: pd.DataFrame, path: Path, dpi: int) -> None:
    prob_cols = [PROBABILITY_COLUMNS[label] for label in LABEL_ORDER]
    max_prob = df[prob_cols].max(axis=1).to_numpy(dtype=float)
    correct = (df["true_label"] == df["interaction_label"]).to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(max_prob[correct], bins=bins, alpha=0.55, density=True, label="correct", color="#17a673")
    ax.hist(max_prob[~correct], bins=bins, alpha=0.55, density=True, label="incorrect", color="#d6336c")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Maximum class probability")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    _save(fig, path, dpi)


def export_evaluation_graphs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = EvaluationGraphConfig(**(config or {}))
    cfg.predictions_path = Path(cfg.predictions_path)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.metadata_path = Path(cfg.metadata_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    _style_plots()
    df = _load_predictions(cfg.predictions_path)
    y_true = df["true_label"].to_numpy()
    y_pred = df["interaction_label"].to_numpy()
    probabilities = df[[PROBABILITY_COLUMNS[label] for label in LABEL_ORDER]]
    malicious_score = df["malicious_probability"].to_numpy(dtype=float)
    selected_threshold = _load_metadata_threshold(cfg.metadata_path)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    paths = {
        "confusion_matrix": cfg.output_dir / "confusion_matrix.png",
        "normalized_confusion_matrix": cfg.output_dir / "normalized_confusion_matrix.png",
        "class_metrics": cfg.output_dir / "class_metrics.png",
        "roc_curves": cfg.output_dir / "roc_curves.png",
        "precision_recall_curves": cfg.output_dir / "precision_recall_curves.png",
        "malicious_threshold_tradeoff": cfg.output_dir / "malicious_threshold_tradeoff.png",
        "malicious_score_distribution": cfg.output_dir / "malicious_score_distribution.png",
        "malicious_calibration": cfg.output_dir / "malicious_calibration.png",
        "confidence_distribution": cfg.output_dir / "confidence_distribution.png",
    }

    _plot_confusion_matrix(y_true, y_pred, paths["confusion_matrix"], cfg.dpi, normalize=False)
    _plot_confusion_matrix(y_true, y_pred, paths["normalized_confusion_matrix"], cfg.dpi, normalize=True)
    _plot_class_metric_bars(report, paths["class_metrics"], cfg.dpi)
    roc_auc_by_class = _plot_roc_curves(y_true, probabilities, paths["roc_curves"], cfg.dpi)
    average_precision_by_class = _plot_precision_recall_curves(y_true, probabilities, paths["precision_recall_curves"], cfg.dpi)
    threshold_summary = _plot_malicious_threshold_tradeoff(
        y_true,
        malicious_score,
        paths["malicious_threshold_tradeoff"],
        cfg.dpi,
        selected_threshold,
    )
    _plot_score_distribution(df, paths["malicious_score_distribution"], cfg.dpi)
    _plot_malicious_calibration(y_true, malicious_score, paths["malicious_calibration"], cfg.dpi)
    _plot_confidence_distribution(df, paths["confidence_distribution"], cfg.dpi)

    label_distribution = df["true_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    pred_distribution = df["interaction_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    manifest = {
        "status": "created",
        "predictions_path": str(cfg.predictions_path.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "rows": int(len(df)),
        "label_order": LABEL_ORDER,
        "label_distribution": {label: int(count) for label, count in label_distribution.items()},
        "prediction_distribution": {label: int(count) for label, count in pred_distribution.items()},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "roc_auc_by_class": roc_auc_by_class,
        "average_precision_by_class": average_precision_by_class,
        "threshold_summary": threshold_summary,
        "graphs": {name: str(path.resolve()) for name, path in paths.items()},
        "config": {
            **asdict(cfg),
            "predictions_path": str(cfg.predictions_path),
            "output_dir": str(cfg.output_dir),
            "metadata_path": str(cfg.metadata_path),
        },
    }
    manifest_path = cfg.output_dir / "evaluation_graphs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest
