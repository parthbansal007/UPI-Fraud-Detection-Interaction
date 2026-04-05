from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.parsers.expat import model

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .feature_matrices import generate_transformer_feature_matrices


@dataclass
class IsolationForestConfig:
    matrix_dir: Path = Path("outputs/processed_matrices")
    data_dir: Path = Path("interaction_data")
    transformer_dir: Path = Path("models/interaction/transformer")
    output_model_path: Path = Path("models/interaction/isolation_forest.joblib")
    output_scores_path: Path = Path("outputs/isolation_forest_scores.csv")
    output_report_path: Path = Path("models/interaction/isolation_forest_report.json")
    random_state: int = 42
    n_estimators: int = 300
    contamination: float = 0.18
    n_jobs: int = -1
    normalize_low_quantile: float = 0.05
    normalize_high_quantile: float = 0.95


def _load_matrices(cfg: IsolationForestConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    required = {"X_train", "X_val", "X_test"}
    missing = sorted(required - set(bundle.files))
    if missing:
        raise ValueError(f"feature_matrices.npz missing required arrays: {missing}")

    return (
        bundle["X_train"].astype(np.float32),
        bundle["X_val"].astype(np.float32),
        bundle["X_test"].astype(np.float32),
    )


def _normalize(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    score = (raw - low) / (high - low)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _default_row_index(train_rows: int, val_rows: int, test_rows: int) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"split": "train", "row_in_split": np.arange(train_rows, dtype=np.int64)}),
            pd.DataFrame({"split": "val", "row_in_split": np.arange(val_rows, dtype=np.int64)}),
            pd.DataFrame({"split": "test_ood", "row_in_split": np.arange(test_rows, dtype=np.int64)}),
        ],
        axis=0,
        ignore_index=True,
    )


def train_isolation_forest_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = IsolationForestConfig(**(config or {}))
    cfg.matrix_dir = Path(cfg.matrix_dir)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.transformer_dir = Path(cfg.transformer_dir)
    cfg.output_model_path = Path(cfg.output_model_path)
    cfg.output_scores_path = Path(cfg.output_scores_path)
    cfg.output_report_path = Path(cfg.output_report_path)

    X_train, X_val, X_test = _load_matrices(cfg)

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    # train only on normal samples (better anomaly detection)
# assume y_train exists or use threshold filtering
    model.fit(X_train[: int(len(X_train) * 0.7)])

    raw_train = -model.decision_function(X_train)
    raw_val = -model.decision_function(X_val)
    raw_test = -model.decision_function(X_test)

    low = float(np.quantile(raw_train, cfg.normalize_low_quantile))
    high = float(np.quantile(raw_train, cfg.normalize_high_quantile))
    if np.isclose(low, high):
        high = low + 1.0

    score_train = _normalize(raw_train, low, high)
    score_val = _normalize(raw_val, low, high)
    score_test = _normalize(raw_test, low, high)

    score_values = pd.DataFrame(
        {
            "anomaly_score_raw": np.concatenate([raw_train, raw_val, raw_test], axis=0),
            "anomaly_score": np.concatenate([score_train, score_val, score_test], axis=0),
        }
    )

    row_index_path = cfg.matrix_dir / "row_index.csv"
    if row_index_path.exists():
        row_index = pd.read_csv(row_index_path)
        if len(row_index) == len(score_values):
            score_df = pd.concat([row_index.reset_index(drop=True), score_values], axis=1)
        else:
            fallback_index = _default_row_index(len(score_train), len(score_val), len(score_test))
            score_df = pd.concat([fallback_index, score_values], axis=1)
    else:
        fallback_index = _default_row_index(len(score_train), len(score_val), len(score_test))
        score_df = pd.concat([fallback_index, score_values], axis=1)

    cfg.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.output_model_path)
    score_df.to_csv(cfg.output_scores_path, index=False)

    report = {
        "fit_data": "train_only",
        "split_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "normalization": {
            "method": "quantile_minmax",
            "low_quantile": float(cfg.normalize_low_quantile),
            "high_quantile": float(cfg.normalize_high_quantile),
            "scale_low": low,
            "scale_high": high,
            "score_range_train": [float(score_train.min()), float(score_train.max())],
            "score_range_val": [float(score_val.min()), float(score_val.max())],
            "score_range_test": [float(score_test.min()), float(score_test.max())],
        },
        "artifact_paths": {
            "isolation_forest_model": str(cfg.output_model_path.resolve()),
            "anomaly_scores": str(cfg.output_scores_path.resolve()),
            "report": str(cfg.output_report_path.resolve()),
            "matrix_bundle": str((cfg.matrix_dir / "feature_matrices.npz").resolve()),
        },
        "config": {
            **asdict(cfg),
            "matrix_dir": str(cfg.matrix_dir),
            "data_dir": str(cfg.data_dir),
            "transformer_dir": str(cfg.transformer_dir),
            "output_model_path": str(cfg.output_model_path),
            "output_scores_path": str(cfg.output_scores_path),
            "output_report_path": str(cfg.output_report_path),
        },
    }
    cfg.output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
