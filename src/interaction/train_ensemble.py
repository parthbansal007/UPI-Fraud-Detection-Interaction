from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

LABEL_ORDER = ["normal", "suspicious", "malicious"]


@dataclass
class EnsembleConfig:
    matrix_dir: Path = Path("outputs/processed_matrices")
    transformer_scores_path: Path = Path("outputs/transformer_probability_scores.csv")
    xgboost_model_path: Path = Path("models/interaction/xgboost_model.json")
    anomaly_scores_path: Path = Path("outputs/isolation_forest_scores.csv")
    output_scores_path: Path = Path("outputs/ensemble_fraud_probabilities.csv")
    output_report_path: Path = Path("models/interaction/ensemble_report.json")
    tune_step: float = 0.05
    decision_threshold: float = 0.5
    default_weights: tuple[float, float, float] = (0.35, 0.55, 0.1)
    malicious_threshold_min: float = 0.50
    malicious_threshold_max: float = 0.75
    suspicious_threshold_min: float = 0.10
    suspicious_threshold_max: float = 0.45
    threshold_step: float = 0.01
    min_malicious_precision: float = 0.75
    default_thresholds: tuple[float, float] = (0.45, 0.35)
    max_weight_candidates_for_threshold_search: int = 60


def _weight_candidates(step: float) -> list[tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("tune_step must be in (0, 1].")
    step_units = int(round(1.0 / step))
    weights: list[tuple[float, float, float]] = []
    for a in range(step_units + 1):
        for b in range(step_units + 1 - a):
            c = step_units - a - b
            w_t = round(a * step, 10)
            w_x = round(b * step, 10)
            w_a = round(c * step, 10)
            weights.append((w_t, w_x, w_a))
    return weights


def _threshold_candidates(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold_step must be > 0.")
    if min_value > max_value:
        raise ValueError("Invalid threshold range: min > max.")
    values: list[float] = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(round(float(current), 6))
        current += step
    return values


def _binary_malicious_metrics(y_true_binary: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "malicious_precision": float(precision_score(y_true_binary, y_pred, zero_division=0)),
        "malicious_recall": float(recall_score(y_true_binary, y_pred, zero_division=0)),
        "malicious_f1": float(f1_score(y_true_binary, y_pred, zero_division=0)),
    }


def _load_inputs(cfg: EnsembleConfig) -> pd.DataFrame:
    row_index_path = cfg.matrix_dir / "row_index.csv"
    matrix_bundle_path = cfg.matrix_dir / "feature_matrices.npz"
    if not row_index_path.exists():
        raise FileNotFoundError(f"Missing row index for matrices: {row_index_path}")
    if not matrix_bundle_path.exists():
        raise FileNotFoundError(f"Missing feature matrices bundle: {matrix_bundle_path}")
    if not cfg.transformer_scores_path.exists():
        raise FileNotFoundError(f"Missing transformer probabilities: {cfg.transformer_scores_path}")
    if not cfg.anomaly_scores_path.exists():
        raise FileNotFoundError(f"Missing anomaly scores: {cfg.anomaly_scores_path}")
    if not cfg.xgboost_model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost model: {cfg.xgboost_model_path}")

    base = pd.read_csv(row_index_path)
    if not {"split", "session_id", "label"}.issubset(base.columns):
        raise ValueError("row_index.csv must contain split, session_id, label.")

    trf = pd.read_csv(cfg.transformer_scores_path)
    needed_trf = {
        "split",
        "session_id",
        "normal_probability",
        "suspicious_probability",
        "malicious_probability",
    }
    if not needed_trf.issubset(trf.columns):
        raise ValueError("Transformer score file missing required columns.")
    trf = trf[list(needed_trf)].rename(
        columns={
            "normal_probability": "transformer_normal_probability",
            "suspicious_probability": "transformer_suspicious_probability",
            "malicious_probability": "transformer_malicious_probability",
        }
    )

    ano = pd.read_csv(cfg.anomaly_scores_path)
    needed_ano = {"split", "session_id", "anomaly_score"}
    if not needed_ano.issubset(ano.columns):
        raise ValueError("Anomaly score file missing required columns.")
    ano = ano[list(needed_ano)]

    bundle = np.load(matrix_bundle_path)
    X = np.vstack([bundle["X_train"], bundle["X_val"], bundle["X_test"]]).astype(np.float32)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(cfg.xgboost_model_path))
    xgb_probs = xgb_model.predict_proba(X)
    if xgb_probs.shape[1] < 3:
        raise ValueError("XGBoost probabilities must include 3 class outputs.")

    out = base.copy()
    out["xgboost_normal_probability"] = xgb_probs[:, 0]
    out["xgboost_suspicious_probability"] = xgb_probs[:, 1]
    out["xgboost_malicious_probability"] = xgb_probs[:, 2]
    out = out.merge(trf, on=["split", "session_id"], how="left")
    out = out.merge(ano, on=["split", "session_id"], how="left")

    missing = {
        "transformer_normal_probability": int(out["transformer_normal_probability"].isna().sum()),
        "transformer_suspicious_probability": int(out["transformer_suspicious_probability"].isna().sum()),
        "transformer_malicious_probability": int(out["transformer_malicious_probability"].isna().sum()),
        "anomaly_score": int(out["anomaly_score"].isna().sum()),
    }
    invalid = {k: v for k, v in missing.items() if v > 0}
    if invalid:
        raise ValueError(f"Missing required ensemble inputs after merge: {invalid}")

    return out


def _build_weighted_scores(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.DataFrame:
    w_t, w_x, w_a = weights
    out = df.copy()
    # normalize anomaly score → make it probability-like
    anom = (out["anomaly_score"] - out["anomaly_score"].min()) / (
        out["anomaly_score"].max() - out["anomaly_score"].min() + 1e-8
    )
    out["malicious_score"] = (
        (w_t * out["transformer_malicious_probability"])
        + (w_x * out["xgboost_malicious_probability"])
        + (w_a * anom)
    )
    out["suspicious_score"] = (
        (w_t * out["transformer_suspicious_probability"])
        + (w_x * out["xgboost_suspicious_probability"])
    ).clip(0.0, 1.0)
    out["normal_score"] = (
        (w_t * out["transformer_normal_probability"])
        + (w_x * out["xgboost_normal_probability"])
        + (w_a * (1.0 - out["anomaly_score"]))
    ).clip(0.0, 1.0)
    out["final_fraud_probability"] = out["malicious_score"]
    total = out["normal_score"] + out["suspicious_score"] + out["malicious_score"] + 1e-8
    out["normal_score"] /= total
    out["suspicious_score"] /= total
    out["malicious_score"] /= total
    return out


def _predict_labels_with_thresholds(
    df: pd.DataFrame,
    malicious_threshold: float,
    suspicious_threshold: float,
) -> np.ndarray:
    labels = np.array(["normal"] * len(df), dtype=object)
    mal_mask = df["malicious_score"].to_numpy(dtype=np.float32) >= malicious_threshold
    labels[mal_mask] = "malicious"

    suspicious_mask = (~mal_mask) & (df["suspicious_score"].to_numpy(dtype=np.float32) >= suspicious_threshold)
    labels[suspicious_mask] = "suspicious"
    return labels


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    y_true_bin = (y_true == "malicious").astype(int)
    y_pred_bin = (y_pred == "malicious").astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "malicious_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "malicious_recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "malicious_f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
        "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
    }


def _false_positive_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    malicious_fp_mask = np.logical_and(y_pred == "malicious", y_true != "malicious")
    normal_overflag_mask = np.logical_and(y_true == "normal", y_pred != "normal")

    by_true_label: dict[str, int] = {}
    if malicious_fp_mask.any():
        uniq, cnt = np.unique(y_true[malicious_fp_mask], return_counts=True)
        by_true_label = {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}

    return {
        "malicious_false_positive_count": int(malicious_fp_mask.sum()),
        "malicious_false_positive_by_true_label": by_true_label,
        "normal_over_flagged_count": int(normal_overflag_mask.sum()),
        "normal_over_flagged_rate": float(normal_overflag_mask.sum() / max(int((y_true == "normal").sum()), 1)),
    }


def train_ensemble_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = EnsembleConfig(**(config or {}))
    cfg.matrix_dir = Path(cfg.matrix_dir)
    cfg.transformer_scores_path = Path(cfg.transformer_scores_path)
    cfg.xgboost_model_path = Path(cfg.xgboost_model_path)
    cfg.anomaly_scores_path = Path(cfg.anomaly_scores_path)
    cfg.output_scores_path = Path(cfg.output_scores_path)
    cfg.output_report_path = Path(cfg.output_report_path)

    data = _load_inputs(cfg)
    val_df = data[data["split"] == "val"].reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Validation split is empty; cannot tune ensemble settings.")
    y_val_binary = (val_df["label"].astype(str).str.lower() == "malicious").astype(int).to_numpy()

    default_w = tuple(float(x) for x in cfg.default_weights)
    if len(default_w) != 3:
        raise ValueError("default_weights must have exactly three values.")
    if not np.isclose(sum(default_w), 1.0):
        raise ValueError("default_weights must sum to 1.0.")

    candidate_weights = _weight_candidates(cfg.tune_step)
    candidate_weights = [w for w in candidate_weights if w[2] <= 0.35]
    if default_w not in candidate_weights:
        candidate_weights.append(default_w)

    weight_results: list[dict[str, Any]] = []

    for w_t, w_x, w_a in candidate_weights:
        weighted_val = _build_weighted_scores(val_df, (w_t, w_x, w_a))
        metrics = _binary_malicious_metrics(
            y_val_binary,
            weighted_val["malicious_score"].to_numpy(dtype=np.float32),
            cfg.decision_threshold,
        )
        weight_results.append(
            {
                "w_transformer": float(w_t),
                "w_xgboost": float(w_x),
                "w_anomaly": float(w_a),
                **metrics,
            }
        )

    top_weight_candidates = sorted(
        weight_results,
        key=lambda r: (r["malicious_f1"], r["malicious_precision"], r["malicious_recall"]),
        reverse=True,
    )[: max(1, int(cfg.max_weight_candidates_for_threshold_search))]
    threshold_weight_candidates = {
        (float(item["w_transformer"]), float(item["w_xgboost"]), float(item["w_anomaly"]))
        for item in top_weight_candidates
    }
    threshold_weight_candidates.add(default_w)

    y_val_labels = val_df["label"].astype(str).str.lower().to_numpy()
    y_val_is_malicious = y_val_labels == "malicious"
    y_val_is_not_malicious = ~y_val_is_malicious
    y_val_is_normal = y_val_labels == "normal"

    default_mal_thr, default_susp_thr = tuple(float(v) for v in cfg.default_thresholds)
    weighted_val_default = _build_weighted_scores(val_df, default_w)
    default_val_pred = _predict_labels_with_thresholds(weighted_val_default, default_mal_thr, default_susp_thr)
    default_threshold_metrics = _multiclass_metrics(y_val_labels, default_val_pred)
    default_fp_summary = _false_positive_summary(y_val_labels, default_val_pred)

    mal_thresholds = _threshold_candidates(cfg.malicious_threshold_min, cfg.malicious_threshold_max, cfg.threshold_step)
    susp_thresholds = _threshold_candidates(cfg.suspicious_threshold_min, cfg.suspicious_threshold_max, cfg.threshold_step)

    threshold_results: list[dict[str, Any]] = []
    best_selection: dict[str, Any] | None = None
    found_meeting_precision = False

    for w_t, w_x, w_a in sorted(threshold_weight_candidates):
        weighted_val = _build_weighted_scores(val_df, (w_t, w_x, w_a))
        mal_scores = weighted_val["malicious_score"].to_numpy(dtype=np.float32)
        susp_scores = weighted_val["suspicious_score"].to_numpy(dtype=np.float32)

        for mal_thr in mal_thresholds:
            mal_pred = mal_scores >= float(mal_thr)

            tp = int(np.logical_and(y_val_is_malicious, mal_pred).sum())
            fp = int(np.logical_and(y_val_is_not_malicious, mal_pred).sum())
            fn = int(np.logical_and(y_val_is_malicious, ~mal_pred).sum())
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

            for susp_thr in susp_thresholds:
                pred = np.where(mal_pred, "malicious", np.where(susp_scores >= float(susp_thr), "suspicious", "normal"))
                normal_overflag_count = int(np.logical_and(y_val_is_normal, pred != "normal").sum())
                precision_ok = precision >= cfg.min_malicious_precision
                if precision_ok:
                    found_meeting_precision = True

                candidate = {
                    "w_transformer": float(w_t),
                    "w_xgboost": float(w_x),
                    "w_anomaly": float(w_a),
                    "malicious_threshold": float(mal_thr),
                    "suspicious_threshold": float(susp_thr),
                    "meets_precision_target": bool(precision_ok),
                    "malicious_precision": precision,
                    "malicious_recall": recall,
                    "malicious_f1": f1,
                    "normal_over_flagged_count": int(normal_overflag_count),
                    "malicious_false_positive_count": int(fp),
                }
                threshold_results.append(candidate)

                if best_selection is None:
                    best_selection = candidate
                    continue

                if found_meeting_precision:
                    if not best_selection["meets_precision_target"] and precision_ok:
                        best_selection = candidate
                        continue
                    if not precision_ok:
                        continue

                    current_key = (
                        -candidate["malicious_f1"],
                        -candidate["malicious_recall"],
                        -candidate["malicious_precision"],
                        candidate["normal_over_flagged_count"],
                        candidate["malicious_false_positive_count"],
                    )
                    best_key = (
                        -best_selection["malicious_f1"],
                        -best_selection["malicious_recall"],
                        -best_selection["malicious_precision"],
                        best_selection["normal_over_flagged_count"],
                        best_selection["malicious_false_positive_count"],
                    )
                    if current_key < best_key:
                        best_selection = candidate
                else:
                    current_key = (
                        -candidate["malicious_f1"],
                        -candidate["malicious_precision"],
                        -candidate["malicious_recall"],
                        candidate["normal_over_flagged_count"],
                        candidate["malicious_false_positive_count"],
                    )
                    best_key = (
                        -best_selection["malicious_f1"],
                        -best_selection["malicious_precision"],
                        -best_selection["malicious_recall"],
                        best_selection["normal_over_flagged_count"],
                        best_selection["malicious_false_positive_count"],
                    )
                    if current_key < best_key:
                        best_selection = candidate

    if best_selection is None:
        raise RuntimeError("Ensemble threshold search did not produce any valid candidate.")

    best_weights = (
        float(best_selection["w_transformer"]),
        float(best_selection["w_xgboost"]),
        float(best_selection["w_anomaly"]),
    )
    best_mal_thr = float(best_selection["malicious_threshold"])
    best_susp_thr = float(best_selection["suspicious_threshold"])

    weighted_all = _build_weighted_scores(data, best_weights)
    weighted_all["predicted_label"] = _predict_labels_with_thresholds(weighted_all, best_mal_thr, best_susp_thr)
    weighted_all["predicted_malicious"] = (weighted_all["predicted_label"] == "malicious").astype(int)

    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "val", "test_ood"]:
        split_df = weighted_all[weighted_all["split"] == split_name]
        y_true = split_df["label"].astype(str).str.lower().to_numpy()
        y_pred = split_df["predicted_label"].astype(str).to_numpy()
        split_metrics[split_name] = _multiclass_metrics(y_true, y_pred)
    tuned_val_df = weighted_all[weighted_all["split"] == "val"].reset_index(drop=True)
    tuned_val_labels = tuned_val_df["label"].astype(str).str.lower().to_numpy()
    tuned_val_pred = tuned_val_df["predicted_label"].astype(str).to_numpy()
    tuned_fp_summary = _false_positive_summary(tuned_val_labels, tuned_val_pred)

    cfg.output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    weighted_all[
        [
            "split",
            "session_id",
            "label",
            "transformer_normal_probability",
            "transformer_suspicious_probability",
            "transformer_malicious_probability",
            "xgboost_normal_probability",
            "xgboost_suspicious_probability",
            "xgboost_malicious_probability",
            "anomaly_score",
            "normal_score",
            "suspicious_score",
            "malicious_score",
            "final_fraud_probability",
            "predicted_label",
            "predicted_malicious",
        ]
    ].to_csv(cfg.output_scores_path, index=False)

    report = {
        "formula": "final_fraud_probability = w_transformer * transformer_malicious + w_xgboost * xgboost_malicious + w_anomaly * anomaly",
        "classification_rule": "no argmax: malicious if malicious_score >= T_m, else suspicious if suspicious_score >= T_s, else normal",
        "baseline_weights": {
            "w_transformer": float(default_w[0]),
            "w_xgboost": float(default_w[1]),
            "w_anomaly": float(default_w[2]),
        },
        "tuned_weights": {
            "w_transformer": float(best_weights[0]),
            "w_xgboost": float(best_weights[1]),
            "w_anomaly": float(best_weights[2]),
        },
        "weight_tuning_objective": "validation malicious F1-score (binary malicious vs rest, coarse search)",
        "best_weight_val_metrics": {
            "malicious_precision": float(best_selection["malicious_precision"]),
            "malicious_recall": float(best_selection["malicious_recall"]),
            "malicious_f1": float(best_selection["malicious_f1"]),
        },
        "baseline_thresholds": {
            "malicious_threshold": float(default_mal_thr),
            "suspicious_threshold": float(default_susp_thr),
        },
        "baseline_threshold_val_metrics": default_threshold_metrics,
        "baseline_false_positive_analysis_val": default_fp_summary,
        "tuned_thresholds": {
            "malicious_threshold": float(best_mal_thr),
            "suspicious_threshold": float(best_susp_thr),
        },
        "threshold_tuning": {
            "optimize_for": "reduce false positives with high malicious precision constraint",
            "precision_target": float(cfg.min_malicious_precision),
            "malicious_threshold_range": [float(cfg.malicious_threshold_min), float(cfg.malicious_threshold_max)],
            "suspicious_threshold_range": [float(cfg.suspicious_threshold_min), float(cfg.suspicious_threshold_max)],
            "threshold_step": float(cfg.threshold_step),
            "num_combinations": int(len(threshold_weight_candidates) * len(mal_thresholds) * len(susp_thresholds)),
            "num_weight_candidates": int(len(threshold_weight_candidates)),
            "met_precision_target": bool(found_meeting_precision),
            "best_val_metrics": {
                "malicious_precision": float(best_selection["malicious_precision"]),
                "malicious_recall": float(best_selection["malicious_recall"]),
                "malicious_f1": float(best_selection["malicious_f1"]),
                "normal_over_flagged_count": int(best_selection["normal_over_flagged_count"]),
                "malicious_false_positive_count": int(best_selection["malicious_false_positive_count"]),
            },
        },
        "tuned_false_positive_analysis_val": tuned_fp_summary,
        "split_metrics": split_metrics,
        "artifact_paths": {
            "ensemble_scores": str(cfg.output_scores_path.resolve()),
            "ensemble_report": str(cfg.output_report_path.resolve()),
            "transformer_scores": str(cfg.transformer_scores_path.resolve()),
            "xgboost_model": str(cfg.xgboost_model_path.resolve()),
            "anomaly_scores": str(cfg.anomaly_scores_path.resolve()),
        },
        "top_weight_candidates": sorted(
            weight_results,
            key=lambda r: (r["malicious_f1"], r["malicious_precision"], r["malicious_recall"]),
            reverse=True,
        )[:10],
        "top_threshold_candidates": sorted(
            threshold_results,
            key=lambda r: (
                int(r["meets_precision_target"]),
                -r["normal_over_flagged_count"],
                -r["malicious_false_positive_count"],
                r["malicious_precision"],
                r["malicious_recall"],
                r["malicious_f1"],
            ),
            reverse=True,
        )[:20],
        "config": {
            **asdict(cfg),
            "matrix_dir": str(cfg.matrix_dir),
            "transformer_scores_path": str(cfg.transformer_scores_path),
            "xgboost_model_path": str(cfg.xgboost_model_path),
            "anomaly_scores_path": str(cfg.anomaly_scores_path),
            "output_scores_path": str(cfg.output_scores_path),
            "output_report_path": str(cfg.output_report_path),
            "default_weights": list(cfg.default_weights),
            "default_thresholds": list(cfg.default_thresholds),
            "max_weight_candidates_for_threshold_search": int(cfg.max_weight_candidates_for_threshold_search),
        },
    }
    cfg.output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
