"""Build cross-model comparison plots for the UPI fraud research report.

Reads the per-model metrics already produced by the trained pipelines and renders
grouped bar charts so the interaction sub-models (and, when available, the
transaction sub-models and unified fusion) can be compared side by side.

Run:
    python analysis/build_comparison_plots.py
Outputs:
    outputs/comparison/*.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grouped_bar(
    title: str,
    models: list[str],
    metrics: dict[str, list[float]],
    out_path: Path,
) -> None:
    n_models = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_models), 5.5))
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)
        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_interaction_comparison() -> None:
    report = _load(ROOT / "models" / "interaction" / "final_project_report.json")
    q = report["quality_summary"]

    # Model -> (accuracy, macro/weighted proxy via malicious_f1, malicious_precision, malicious_recall)
    rows = {
        "Transformer\n(DistilBERT)": q["transformer_test_ood"],
        "XGBoost": q["xgboost_test_ood"],
        "Weighted\nEnsemble": q["ensemble_test_ood"],
        "Runtime\nCalibrated": q["calibrated_runtime_test_ood"],
        "Hybrid\n(train-time)": q["historical_training_test_ood"],
    }

    def mal_f1(d: dict[str, Any]) -> float:
        if "malicious_f1" in d:
            return d["malicious_f1"]
        return d["f1_by_class"]["malicious"]

    models = list(rows.keys())
    metrics = {
        "Accuracy": [rows[m]["accuracy"] for m in models],
        "Malicious Precision": [rows[m]["malicious_precision"] for m in models],
        "Malicious Recall": [rows[m]["malicious_recall"] for m in models],
        "Malicious F1": [mal_f1(rows[m]) for m in models],
    }
    _grouped_bar(
        "Interaction Model — Sub-model Comparison (test_ood, 1500 sessions)",
        models,
        metrics,
        OUT_DIR / "interaction_model_comparison.png",
    )


def build_transaction_comparison() -> None:
    path = ROOT / "models" / "transaction" / "transaction_training_report.json"
    if not path.exists():
        print(f"skip transaction comparison (missing {path})")
        return
    report = _load(path)
    per_model = report.get("results") or report.get("model_metrics") or report.get("models") or {}
    if not per_model:
        print("skip transaction comparison (no per-model metrics in report)")
        return
    label_map = {"logistic": "Logistic\nRegression", "rf": "Random\nForest", "xgboost": "XGBoost"}
    keys = list(per_model.keys())
    models = [label_map.get(k, k) for k in keys]

    def g(d: dict[str, Any], *names: str) -> float:
        for n in names:
            if n in d:
                return float(d[n])
        return 0.0

    metrics = {
        "Precision (fraud)": [g(per_model[k], "precision") for k in keys],
        "Recall (fraud)": [g(per_model[k], "recall") for k in keys],
        "F1 (fraud)": [g(per_model[k], "f1") for k in keys],
        "PR-AUC": [g(per_model[k], "pr_auc") for k in keys],
        "ROC-AUC": [g(per_model[k], "roc_auc", "auc") for k in keys],
    }
    _grouped_bar(
        "Transaction Model — Classifier Comparison (held-out test, fraud rate 0.19%)",
        models,
        metrics,
        OUT_DIR / "transaction_model_comparison.png",
    )


def build_unified_comparison() -> None:
    inter = _load(ROOT / "outputs" / "interaction" / "metrics.json")
    uni_path = ROOT / "outputs" / "unified" / "metrics.json"
    txn_path = ROOT / "outputs" / "transaction" / "metrics.json"
    if not uni_path.exists():
        print(f"skip unified comparison (missing {uni_path})")
        return
    uni = _load(uni_path)
    series: dict[str, dict[str, float]] = {}

    def take(d: dict[str, Any]) -> dict[str, float]:
        return {
            "Accuracy": float(d.get("accuracy", 0.0)),
            "ROC-AUC": float(d.get("roc_auc", 0.0)),
            "Malicious/Fraud Precision": float(
                d.get("malicious_precision", d.get("fraud_precision", d.get("precision", 0.0)))
            ),
            "Malicious/Fraud Recall": float(
                d.get("malicious_recall", d.get("fraud_recall", d.get("recall", 0.0)))
            ),
        }

    series["Interaction\nonly"] = take(inter)
    if txn_path.exists():
        series["Transaction\nonly"] = take(_load(txn_path))
    series["Unified\nFusion"] = take(uni)

    models = list(series.keys())
    metric_names = list(next(iter(series.values())).keys())
    metrics = {mn: [series[m][mn] for m in models] for mn in metric_names}
    _grouped_bar(
        "Unified Fusion vs Individual Models",
        models,
        metrics,
        OUT_DIR / "unified_model_comparison.png",
    )


def main() -> None:
    build_interaction_comparison()
    build_transaction_comparison()
    build_unified_comparison()


if __name__ == "__main__":
    main()
