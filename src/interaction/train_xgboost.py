from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

from .feature_matrices import generate_transformer_feature_matrices

LABEL_ORDER = ["normal", "suspicious", "malicious"]


@dataclass
class XGBoostTuningConfig:
    matrix_dir: Path = Path("outputs/processed_matrices")
    data_dir: Path = Path("interaction_data")
    transformer_dir: Path = Path("models/interaction/transformer")
    output_model_path: Path = Path("models/interaction/xgboost_model.json")
    output_report_path: Path = Path("models/interaction/xgboost_tuning_report.json")
    random_state: int = 42
    tree_method: str = "hist"
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    class_weights: tuple[float, float, float] = (1.0, 4.0, 7.0)
    max_depth_grid: tuple[int, ...] = (4, 6, 8)
    learning_rate_grid: tuple[float, ...] = (0.01, 0.05, 0.1)
    n_estimators_grid: tuple[int, ...] = (200, 400, 600)


def _metrics_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    y_bin = (y_true == 2).astype(int)
    p_bin = (y_pred == 2).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "malicious_precision": float(precision_score(y_bin, p_bin, zero_division=0)),
        "malicious_recall": float(recall_score(y_bin, p_bin, zero_division=0)),
        "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
        "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
    }


def _load_matrices(cfg: XGBoostTuningConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle_path = cfg.matrix_dir / "feature_matrices.npz"
    if not bundle_path.exists():
        generate_transformer_feature_matrices(
            {
                "data_dir": cfg.data_dir,
                "transformer_dir": cfg.transformer_dir,
                "output_dir": cfg.matrix_dir,
                "seed": cfg.random_state,
            }
        )

    bundle = np.load(bundle_path)
    required = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    missing = sorted(required - set(bundle.files))
    if missing:
        raise ValueError(f"feature_matrices.npz missing required arrays: {missing}")

    return (
        bundle["X_train"].astype(np.float32),
        bundle["X_val"].astype(np.float32),
        bundle["X_test"].astype(np.float32),
        bundle["y_train"].astype(np.int64),
        bundle["y_val"].astype(np.int64),
        bundle["y_test"].astype(np.int64),
    )


def _build_sample_weights(y: np.ndarray, class_weights: tuple[float, float, float]) -> np.ndarray:
    if not (class_weights[2] >= class_weights[1] >= class_weights[0]):
        raise ValueError("Class weights must keep malicious highest and normal lowest.")
    mapping = np.array(class_weights, dtype=np.float32)
    return mapping[y]


def train_optimized_xgboost(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = XGBoostTuningConfig(**(config or {}))
    cfg.matrix_dir = Path(cfg.matrix_dir)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.transformer_dir = Path(cfg.transformer_dir)
    cfg.output_model_path = Path(cfg.output_model_path)
    cfg.output_report_path = Path(cfg.output_report_path)

    X_train, X_val, X_test, y_train, y_val, y_test = _load_matrices(cfg)
    train_weights = _build_sample_weights(y_train, cfg.class_weights)

    tuning_results: list[dict[str, Any]] = []
    best_model: xgb.XGBClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_val_precision = -1.0
    best_val_recall = -1.0
    prefer_gpu = bool(torch.cuda.is_available())

    for max_depth, learning_rate, n_estimators in product(
        cfg.max_depth_grid,
        cfg.learning_rate_grid,
        cfg.n_estimators_grid,
    ):
        xgb_kwargs: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": cfg.random_state,
            "tree_method": "hist",
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "gamma": 0.1,
            "n_jobs": cfg.n_jobs,
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "n_estimators": int(n_estimators),
            "early_stopping_rounds": cfg.early_stopping_rounds,
        }
        if prefer_gpu:
            xgb_kwargs["device"] = "cuda"

        model = xgb.XGBClassifier(**xgb_kwargs)
        try:
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        except xgb.core.XGBoostError:
            if not prefer_gpu:
                raise
            prefer_gpu = False
            xgb_kwargs.pop("device", None)
            model = xgb.XGBClassifier(**xgb_kwargs)
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        val_pred = model.predict(X_val)
        val_metrics = _metrics_payload(y_val, val_pred)
        precision = float(val_metrics["malicious_precision"])
        recall = float(val_metrics["malicious_recall"])

        result = {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "n_estimators": int(n_estimators),
            "best_iteration": int(getattr(model, "best_iteration", n_estimators)),
            "val_metrics": val_metrics,
        }
        tuning_results.append(result)

        is_better = precision > best_val_precision or (
            np.isclose(precision, best_val_precision) and recall > best_val_recall
        )
        if is_better:
            best_model = model
            best_params = {
                "max_depth": int(max_depth),
                "learning_rate": float(learning_rate),
                "n_estimators": int(n_estimators),
                "best_iteration": int(getattr(model, "best_iteration", n_estimators)),
            }
            best_val_precision = precision
            best_val_recall = recall

    if best_model is None or best_params is None:
        raise RuntimeError("XGBoost tuning did not produce a valid model.")

    val_pred_best = best_model.predict(X_val)
    test_pred_best = best_model.predict(X_test)
    val_metrics_best = _metrics_payload(y_val, val_pred_best)
    test_metrics_best = _metrics_payload(y_test, test_pred_best)

    cfg.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(cfg.output_model_path))

    report = {
        "target_objective": "maximize malicious precision, break ties by malicious recall",
        "class_weights": {
            "normal": float(cfg.class_weights[0]),
            "suspicious": float(cfg.class_weights[1]),
            "malicious": float(cfg.class_weights[2]),
        },
        "search_grid": {
            "max_depth": list(cfg.max_depth_grid),
            "learning_rate": list(cfg.learning_rate_grid),
            "n_estimators": list(cfg.n_estimators_grid),
        },
        "early_stopping_rounds": int(cfg.early_stopping_rounds),
        "split_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "best_params": best_params,
        "best_val_metrics": val_metrics_best,
        "test_ood_metrics": test_metrics_best,
        "tuning_results": tuning_results,
        "artifact_paths": {
            "xgboost_model_json": str(cfg.output_model_path.resolve()),
            "tuning_report": str(cfg.output_report_path.resolve()),
            "matrix_bundle": str((cfg.matrix_dir / "feature_matrices.npz").resolve()),
        },
        "config": {
            **asdict(cfg),
            "matrix_dir": str(cfg.matrix_dir),
            "data_dir": str(cfg.data_dir),
            "transformer_dir": str(cfg.transformer_dir),
            "output_model_path": str(cfg.output_model_path),
            "output_report_path": str(cfg.output_report_path),
            "class_weights": list(cfg.class_weights),
            "max_depth_grid": list(cfg.max_depth_grid),
            "learning_rate_grid": list(cfg.learning_rate_grid),
            "n_estimators_grid": list(cfg.n_estimators_grid),
        },
    }
    cfg.output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
