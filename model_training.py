from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from common import parse_record, read_dataset
from feature_pipeline import suspicious_subsequence_score

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import shap
except Exception:
    shap = None


RESTRICTED_FEATURE_COLUMNS = {
    "label",
    "label_id",
    "risk_score",
    "split",
    "split_reason",
    "generator_version",
    "scenario_family",
    "session_id",
}

NON_NUMERIC_DROP_COLUMNS = {
    "timestamp_start",
    "timestamp_end",
    "event_sequence_encoded",
}

CLASS_NAMES = ["normal", "suspicious", "malicious"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}
CLASS_WEIGHTS = {0: 1.0, 1: 2.0, 2: 2.5}
RAPID_ACTIVITY_THRESHOLD = 5.0
PERMISSION_REQUEST_THRESHOLD = 3
MALICIOUS_THRESHOLD = 0.5
SUSPICIOUS_THRESHOLD = 0.35
LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6
FOCAL_GAMMA = 1.5
SUSPICIOUS_FOCAL_BOOST = 1.35
MALICIOUS_FOCAL_BOOST = 1.15


@dataclass
class EncodedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]


@dataclass
class FraudDetectionSystem:
    xgb_model: Any
    lgbm_model: Any | None
    feature_columns: list[str]
    class_names: list[str]
    ensemble_weights: dict[str, float]
    class_weights: dict[int, float]
    rapid_activity_threshold: float
    permission_request_threshold: int
    suspicious_threshold: float
    malicious_threshold: float
    low_risk_threshold: float
    high_risk_threshold: float
    global_feature_importance: dict[str, list[dict[str, Any]]]

    def prepare_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            frame = x.copy()
            for column in self.feature_columns:
                if column not in frame.columns:
                    frame[column] = 0.0
            return frame[self.feature_columns].astype(float)
        matrix = np.asarray(x, dtype=float)
        return pd.DataFrame(matrix, columns=self.feature_columns)

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self.prepare_frame(x)
        probabilities = self.ensemble_weights["xgboost"] * self.xgb_model.predict_proba(frame)
        if self.lgbm_model is not None and self.ensemble_weights["lightgbm"] > 0:
            probabilities += self.ensemble_weights["lightgbm"] * self.lgbm_model.predict_proba(frame)
        return probabilities

    def compute_rule_score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self.prepare_frame(x)
        rule_score = np.zeros(len(frame), dtype=float)
        vpn = frame.get("vpn", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        ip_risk = frame.get("ip_risk_score", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        permission_requests = frame.get("num_permission_requests", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        qr_scans = frame.get("num_qr_scans", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        links_clicked = frame.get("num_links_clicked", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        rule_score += ((vpn >= 1.0) & (ip_risk > 0.7)).astype(float) * 0.25
        rule_score += (permission_requests > float(self.permission_request_threshold)).astype(float) * 0.25
        rule_score += ((qr_scans > 0.0) & (links_clicked > 0.0)).astype(float) * 0.35
        return np.clip(rule_score, 0.0, 1.0)

    def score_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = self.prepare_frame(x)
        probabilities = self.predict_proba(frame)
        model_score = probabilities[:, CLASS_TO_INDEX["malicious"]]
        rule_score = self.compute_rule_score(frame)
        final_score = np.clip(model_score + rule_score, 0.0, 1.0)
        predicted_index = np.where(
            final_score > self.malicious_threshold,
            CLASS_TO_INDEX["malicious"],
            np.where(final_score > self.suspicious_threshold, CLASS_TO_INDEX["suspicious"], CLASS_TO_INDEX["normal"]),
        )
        risk_level = np.where(
            final_score < self.low_risk_threshold,
            "LOW RISK",
            np.where(final_score < self.high_risk_threshold, "MEDIUM RISK", "HIGH RISK"),
        )
        return pd.DataFrame(
            {
                "predicted_label": [self.class_names[idx] for idx in predicted_index],
                "predicted_index": predicted_index.astype(int),
                "model_score": model_score.astype(float),
                "rule_score": rule_score.astype(float),
                "final_score": final_score.astype(float),
                "risk_level": risk_level,
                "prob_normal": probabilities[:, CLASS_TO_INDEX["normal"]].astype(float),
                "prob_suspicious": probabilities[:, CLASS_TO_INDEX["suspicious"]].astype(float),
                "prob_malicious": probabilities[:, CLASS_TO_INDEX["malicious"]].astype(float),
            },
            index=frame.index,
        )


def validate_runtime_dependencies() -> None:
    missing = []
    if XGBClassifier is None:
        missing.append("xgboost")
    if shap is None:
        missing.append("shap")
    if missing:
        raise ImportError(f"Missing required training dependencies: {missing}")


def validate_feature_columns(feature_columns: list[str]) -> None:
    forbidden = sorted(set(feature_columns) & RESTRICTED_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden columns were selected as model features: {forbidden}")


def safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.astype(float).div(denominator.astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_divide_scalar(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def parse_encoded_sequence(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def derive_behavioral_features_from_encoded(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    num_events = pd.to_numeric(working.get("num_events", 0), errors="coerce").fillna(0.0)
    if "num_permission_requests" in working.columns:
        working["permission_abuse_ratio"] = safe_divide_series(working["num_permission_requests"], num_events)
    if "num_links_clicked" in working.columns:
        working["link_click_rate"] = safe_divide_series(working["num_links_clicked"], num_events)
    if "num_qr_scans" in working.columns:
        working["qr_to_action_ratio"] = safe_divide_series(working["num_qr_scans"], num_events)
    if "session_duration" in working.columns:
        average_seconds = safe_divide_series(working["session_duration"], num_events)
        working["rapid_activity_flag"] = (average_seconds < RAPID_ACTIVITY_THRESHOLD).astype(float)
        working["avg_seconds_per_event"] = average_seconds.astype(float)
    if "event_sequence_encoded" in working.columns:
        working["event_diversity"] = working["event_sequence_encoded"].apply(lambda value: float(len(set(parse_encoded_sequence(value)))))
    else:
        event_count_columns = sorted(column for column in working.columns if column.startswith("event_count_"))
        if event_count_columns:
            working["event_diversity"] = (working[event_count_columns].fillna(0) > 0).sum(axis=1).astype(float)
    if {"ip_risk_score", "vpn"}.issubset(working.columns):
        ip_risk = pd.to_numeric(working["ip_risk_score"], errors="coerce").fillna(0.0)
        vpn = pd.to_numeric(working["vpn"], errors="coerce").fillna(0.0)
        working["borderline_device_risk"] = (((ip_risk >= 0.4) & (ip_risk <= 0.7)) | ((vpn > 0.0) & (ip_risk >= 0.25))).astype(float)
    if {"num_links_clicked", "num_qr_scans", "num_permission_requests"}.issubset(working.columns):
        working["borderline_interaction_score"] = (
            pd.to_numeric(working["num_links_clicked"], errors="coerce").fillna(0.0) * 0.35
            + pd.to_numeric(working["num_qr_scans"], errors="coerce").fillna(0.0) * 0.3
            + pd.to_numeric(working["num_permission_requests"], errors="coerce").fillna(0.0) * 0.35
        )
        working["mixed_channel_flag"] = (
            (pd.to_numeric(working["num_links_clicked"], errors="coerce").fillna(0.0) > 0.0)
            & (pd.to_numeric(working["num_qr_scans"], errors="coerce").fillna(0.0) > 0.0)
        ).astype(float)
    if {"event_diversity", "num_events"}.issubset(working.columns):
        working["event_diversity_ratio"] = safe_divide_series(working["event_diversity"], num_events)
        working["behavioral_entropy_proxy"] = (
            safe_divide_series(working["event_diversity"], np.sqrt(num_events.clip(lower=1.0)))
        ).astype(float)
    if {"link_click_rate", "permission_abuse_ratio", "qr_to_action_ratio", "event_diversity_ratio"}.issubset(working.columns):
        working["borderline_behavior_score"] = (
            working["link_click_rate"].astype(float) * 0.3
            + working["permission_abuse_ratio"].astype(float) * 0.3
            + working["qr_to_action_ratio"].astype(float) * 0.2
            + working["event_diversity_ratio"].astype(float) * 0.2
        )
    return working


def derive_behavioral_features_from_audit_dataset(dataset_path: str) -> pd.DataFrame:
    raw = read_dataset(dataset_path)
    rows = []
    for _, row in raw.iterrows():
        record = parse_record(row)
        events = record.events
        event_count = float(max(len(events), 1))
        permission_requests = float(record.behavioral_features.get("num_permission_requests", events.count("grant_permission") + events.count("deny_permission")))
        link_clicks = float(record.behavioral_features.get("num_links_clicked", events.count("click_link")))
        qr_scans = float(record.behavioral_features.get("num_qr_scans", events.count("scan_qr")))
        rows.append(
            {
                "session_id": row["session_id"],
                "permission_abuse_ratio": safe_divide_scalar(permission_requests, event_count),
                "link_click_rate": safe_divide_scalar(link_clicks, event_count),
                "qr_to_action_ratio": safe_divide_scalar(qr_scans, event_count),
                "rapid_activity_flag": float(safe_divide_scalar(record.session_duration, event_count) < RAPID_ACTIVITY_THRESHOLD),
                "avg_seconds_per_event": safe_divide_scalar(record.session_duration, event_count),
                "event_diversity": float(len(set(events))),
                "suspicious_pattern_score": suspicious_subsequence_score(events),
                "borderline_device_risk": float(
                    (float(record.device_state.get("ip_risk_score", 0.0) or 0.0) >= 0.4 and float(record.device_state.get("ip_risk_score", 0.0) or 0.0) <= 0.7)
                    or ((float(record.device_state.get("vpn", 0.0) or 0.0) > 0.0) and (float(record.device_state.get("ip_risk_score", 0.0) or 0.0) >= 0.25))
                ),
                "borderline_interaction_score": link_clicks * 0.35 + qr_scans * 0.3 + permission_requests * 0.35,
                "mixed_channel_flag": float(link_clicks > 0.0 and qr_scans > 0.0),
                "event_diversity_ratio": safe_divide_scalar(float(len(set(events))), event_count),
                "behavioral_entropy_proxy": safe_divide_scalar(float(len(set(events))), float(np.sqrt(max(event_count, 1.0)))),
                "borderline_behavior_score": (
                    safe_divide_scalar(link_clicks, event_count) * 0.3
                    + safe_divide_scalar(permission_requests, event_count) * 0.3
                    + safe_divide_scalar(qr_scans, event_count) * 0.2
                    + safe_divide_scalar(float(len(set(events))), event_count) * 0.2
                ),
            }
        )
    return pd.DataFrame(rows)


def load_encoded_dataset(dataset_path: str, audit_dataset_path: str | None = "dataset_v2.csv") -> EncodedDataset:
    frame = pd.read_csv(dataset_path)
    frame = derive_behavioral_features_from_encoded(frame)
    if audit_dataset_path and Path(audit_dataset_path).exists() and "session_id" in frame.columns:
        audit_features = derive_behavioral_features_from_audit_dataset(audit_dataset_path)
        frame = frame.merge(audit_features, on="session_id", how="left", suffixes=("", "_audit"))
        for column in [
            "permission_abuse_ratio",
            "link_click_rate",
            "qr_to_action_ratio",
            "rapid_activity_flag",
            "avg_seconds_per_event",
            "event_diversity",
            "suspicious_pattern_score",
            "borderline_device_risk",
            "borderline_interaction_score",
            "mixed_channel_flag",
            "event_diversity_ratio",
            "behavioral_entropy_proxy",
            "borderline_behavior_score",
        ]:
            audit_column = f"{column}_audit"
            if audit_column in frame.columns:
                frame[column] = frame[audit_column].fillna(frame.get(column, 0.0))
                frame = frame.drop(columns=[audit_column])
    if "label" not in frame.columns and "label_id" in frame.columns:
        frame["label"] = pd.to_numeric(frame["label_id"], errors="coerce")
    required = {"label", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset_encoded_v2.csv is missing required columns: {missing}")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    if frame["label"].isna().any():
        raise ValueError("label column must contain numeric class ids")
    if set(frame["split"].dropna().unique()) != {"train", "val", "test_ood"}:
        raise ValueError("split column must contain exactly train, val, and test_ood")
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("label column must contain only 0, 1, and 2")
    feature_columns: list[str] = []
    for column in frame.columns:
        if column in RESTRICTED_FEATURE_COLUMNS or column in NON_NUMERIC_DROP_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().all():
            continue
        frame[column] = numeric.fillna(numeric.median())
        feature_columns.append(column)
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found after applying restrictions")
    validate_feature_columns(feature_columns)
    frame["label_index"] = frame["label"].astype(int)
    return EncodedDataset(frame=frame, feature_columns=feature_columns)


def split_dataset(encoded: EncodedDataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = encoded.frame
    train_df = frame.loc[frame["split"] == "train"].copy()
    val_df = frame.loc[frame["split"] == "val"].copy()
    test_df = frame.loc[frame["split"] == "test_ood"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more required splits are empty")
    return train_df, val_df, test_df


def sample_weights_from_frame(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    base_weights = np.asarray([CLASS_WEIGHTS[int(label)] for label in labels], dtype=float)
    suspicious_mask = labels == CLASS_TO_INDEX["suspicious"]
    malicious_mask = labels == CLASS_TO_INDEX["malicious"]
    suspicious_pattern = frame.get("suspicious_pattern_score", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    borderline_behavior = frame.get("borderline_behavior_score", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    device_risk = frame.get("borderline_device_risk", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    mixed_channel = frame.get("mixed_channel_flag", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    event_diversity_ratio = frame.get("event_diversity_ratio", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    suspicious_boost = 1.0 + (
        np.clip(borderline_behavior, 0.0, 1.5) * 0.45
        + np.clip(device_risk, 0.0, 1.0) * 0.2
        + np.clip(mixed_channel, 0.0, 1.0) * 0.15
        + np.clip(event_diversity_ratio, 0.0, 1.0) * 0.2
    )
    malicious_boost = 1.0 + np.clip(suspicious_pattern, 0.0, 2.0) * 0.1
    base_weights[suspicious_mask] *= suspicious_boost[suspicious_mask] ** FOCAL_GAMMA * SUSPICIOUS_FOCAL_BOOST
    base_weights[malicious_mask] *= malicious_boost[malicious_mask] ** (FOCAL_GAMMA * 0.5) * MALICIOUS_FOCAL_BOOST
    return base_weights.astype(float)


def build_xgboost_model(params: dict[str, Any] | None = None) -> Any:
    if XGBClassifier is None:
        raise ImportError("xgboost is required for training this fraud detection system")
    defaults = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_NAMES),
        "eval_metric": "mlogloss",
        "n_estimators": 320,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def build_lightgbm_model(params: dict[str, Any] | None = None) -> Any | None:
    if LGBMClassifier is None:
        return None
    defaults = {
        "objective": "multiclass",
        "class_weight": CLASS_WEIGHTS,
        "n_estimators": 280,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def risk_level_from_scores(final_score: np.ndarray) -> list[str]:
    return np.where(
        final_score < LOW_RISK_THRESHOLD,
        "LOW RISK",
        np.where(final_score < HIGH_RISK_THRESHOLD, "MEDIUM RISK", "HIGH RISK"),
    ).tolist()


def humanize_feature(feature: str) -> str:
    if feature in {"permission_abuse_ratio", "num_permission_requests"}:
        return "High permission usage"
    if feature in {"link_click_rate", "num_links_clicked"}:
        return "Suspicious link activity"
    if feature in {"qr_to_action_ratio", "num_qr_scans"}:
        return "QR-linked payment behavior"
    if feature in {"ip_risk_score", "vpn"}:
        return "High IP risk"
    if feature in {"rapid_activity_flag", "session_duration", "num_events"}:
        return "Rapid session activity"
    if feature in {"suspicious_pattern_score", "event_diversity"}:
        return "Fraud-like event sequence"
    if feature in {"borderline_behavior_score", "borderline_interaction_score", "event_diversity_ratio"}:
        return "Borderline suspicious behavior"
    if feature in {"borderline_device_risk", "mixed_channel_flag"}:
        return "Mixed risk signals"
    label = feature.replace("_", " ").replace("::", " ").strip()
    return label[:1].upper() + label[1:] if label else feature


def extract_top_feature_importance(model: Any, feature_columns: list[str], top_k: int = 15) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    values = np.asarray(importances, dtype=float)
    order = np.argsort(values)[::-1][:top_k]
    return [{"feature": feature_columns[idx], "importance": float(values[idx])} for idx in order]


def _malicious_shap_values(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, CLASS_TO_INDEX["malicious"]]
    if isinstance(values, list):
        return np.asarray(values[CLASS_TO_INDEX["malicious"]])
    if array.ndim == 2:
        return array
    raise ValueError("Unsupported SHAP output structure")


def explain_prediction(sample: pd.Series | pd.DataFrame | dict[str, Any], model: FraudDetectionSystem, shap_explainer: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        frame = pd.DataFrame([sample])
    elif isinstance(sample, pd.Series):
        frame = sample.to_frame().T
    else:
        frame = sample.copy()
    prepared = model.prepare_frame(frame)
    shap_values = _malicious_shap_values(shap_explainer.shap_values(prepared))
    contributions = shap_values[0]
    order = np.argsort(np.abs(contributions))[::-1][:3]
    top_features = [
        {
            "feature": model.feature_columns[idx],
            "contribution": float(contributions[idx]),
            "value": float(prepared.iloc[0, idx]),
        }
        for idx in order
    ]
    explanation_text = "; ".join(humanize_feature(item["feature"]) for item in top_features)
    return {
        "top_features": top_features,
        "explanation_text": explanation_text,
    }


def evaluate_hybrid_predictions(y_true: np.ndarray, scored: pd.DataFrame) -> dict[str, Any]:
    predicted = scored["predicted_index"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=[0, 1, 2],
        zero_division=0,
    )
    report = classification_report(
        y_true,
        predicted,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return {
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
        "f1_macro": float(f1_score(y_true, predicted, average="macro")),
        "f1_weighted": float(f1_score(y_true, predicted, average="weighted")),
        "per_class": {
            CLASS_NAMES[idx]: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(len(CLASS_NAMES))
        },
    }


def malicious_priority_score(metrics: dict[str, Any]) -> float:
    malicious = metrics["per_class"]["malicious"]
    suspicious = metrics["per_class"]["suspicious"]
    return (
        malicious["recall"] * 0.5
        + malicious["precision"] * 0.2
        + suspicious["recall"] * 0.2
        + suspicious["f1"] * 0.05
        + metrics["f1_macro"] * 0.05
    )


def train_candidate_system(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_columns: list[str],
    xgb_params: dict[str, Any],
    lgbm_params: dict[str, Any] | None,
    ensemble_weights: dict[str, float],
) -> tuple[FraudDetectionSystem, dict[str, Any]]:
    xgb_model = build_xgboost_model(xgb_params)
    xgb_model.fit(x_train, y_train, sample_weight=sample_weights_from_frame(x_train, y_train))
    lgbm_model = None
    if ensemble_weights["lightgbm"] > 0 and LGBMClassifier is not None:
        lgbm_model = build_lightgbm_model(lgbm_params)
        lgbm_model.fit(x_train, y_train)
    system = FraudDetectionSystem(
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        feature_columns=feature_columns,
        class_names=CLASS_NAMES,
        ensemble_weights=ensemble_weights,
        class_weights=CLASS_WEIGHTS,
        rapid_activity_threshold=RAPID_ACTIVITY_THRESHOLD,
        permission_request_threshold=PERMISSION_REQUEST_THRESHOLD,
        suspicious_threshold=SUSPICIOUS_THRESHOLD,
        malicious_threshold=MALICIOUS_THRESHOLD,
        low_risk_threshold=LOW_RISK_THRESHOLD,
        high_risk_threshold=HIGH_RISK_THRESHOLD,
        global_feature_importance={
            "xgboost": extract_top_feature_importance(xgb_model, feature_columns),
            "lightgbm": extract_top_feature_importance(lgbm_model, feature_columns) if lgbm_model is not None else [],
        },
    )
    scored_val = system.score_frame(x_val)
    metrics = evaluate_hybrid_predictions(y_val, scored_val)
    summary = {
        "xgboost_params": xgb_params,
        "lightgbm_params": lgbm_params if lgbm_model is not None else None,
        "ensemble_weights": ensemble_weights,
        "training_loss_strategy": "focal_style_sample_weighting",
        "malicious_precision": metrics["per_class"]["malicious"]["precision"],
        "malicious_recall": metrics["per_class"]["malicious"]["recall"],
        "suspicious_precision": metrics["per_class"]["suspicious"]["precision"],
        "suspicious_recall": metrics["per_class"]["suspicious"]["recall"],
        "f1_macro": metrics["f1_macro"],
        "score": float(malicious_priority_score(metrics)),
    }
    return system, summary


def tune_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_columns: list[str],
) -> tuple[FraudDetectionSystem, dict[str, Any]]:
    xgb_candidates = [
        {"n_estimators": 240, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 320, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 420, "max_depth": 7, "learning_rate": 0.035, "subsample": 0.95, "colsample_bytree": 0.85},
    ]
    lgbm_candidates = [None]
    if LGBMClassifier is not None:
        lgbm_candidates = [
            {"n_estimators": 240, "learning_rate": 0.06, "num_leaves": 31},
            {"n_estimators": 320, "learning_rate": 0.05, "num_leaves": 63},
        ]
    weight_candidates = [{"xgboost": 1.0, "lightgbm": 0.0}]
    if LGBMClassifier is not None:
        weight_candidates.extend(
            [
                {"xgboost": 0.85, "lightgbm": 0.15},
                {"xgboost": 0.7, "lightgbm": 0.3},
            ]
        )
    best_system = None
    best_summary = None
    trial_summaries: list[dict[str, Any]] = []
    for xgb_params in xgb_candidates:
        candidate_lgbm_params = lgbm_candidates if LGBMClassifier is not None else [None]
        for lgbm_params in candidate_lgbm_params:
            for ensemble_weights in weight_candidates:
                if ensemble_weights["lightgbm"] > 0 and lgbm_params is None:
                    continue
                system, summary = train_candidate_system(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    feature_columns=feature_columns,
                    xgb_params=xgb_params,
                    lgbm_params=lgbm_params,
                    ensemble_weights=ensemble_weights,
                )
                trial_summaries.append(summary)
                if best_summary is None or summary["score"] > best_summary["score"]:
                    best_system = system
                    best_summary = summary
    if best_system is None or best_summary is None:
        raise RuntimeError("Model tuning failed to produce a valid candidate")
    return best_system, {"best_trial": best_summary, "trial_summaries": trial_summaries}


def build_predictions_with_scores(test_df: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    payload = pd.DataFrame(index=test_df.index)
    payload["predicted_label"] = scored["predicted_label"]
    payload["model_score"] = scored["model_score"]
    payload["rule_score"] = scored["rule_score"]
    payload["final_score"] = scored["final_score"]
    payload["risk_level"] = scored["risk_level"]
    payload["true_label"] = test_df["label_index"].map({idx: label for label, idx in CLASS_TO_INDEX.items()})
    payload["split"] = test_df["split"].astype(str)
    if "session_id" in test_df.columns:
        payload["session_id"] = test_df["session_id"].astype(str)
    return payload.reset_index(drop=True)


def generate_sample_explanations(
    test_features: pd.DataFrame,
    scored: pd.DataFrame,
    model: FraudDetectionSystem,
    shap_explainer: Any,
    sample_count: int = 3,
) -> list[dict[str, Any]]:
    order = np.argsort(scored["final_score"].to_numpy(dtype=float))[::-1][:sample_count]
    explanations = []
    for idx in order:
        explanation = explain_prediction(test_features.iloc[[idx]], model, shap_explainer)
        explanations.append(
            {
                "row_index": int(idx),
                "predicted_label": str(scored.iloc[idx]["predicted_label"]),
                "risk_level": str(scored.iloc[idx]["risk_level"]),
                "final_score": float(scored.iloc[idx]["final_score"]),
                "explanation": explanation,
            }
        )
    return explanations


def train_pipeline(
    dataset_path: str = "dataset_encoded_v2.csv",
    audit_dataset_path: str | None = "dataset_v2.csv",
    model_output_path: str = "trained_model.pkl",
    feature_output_path: str = "feature_columns.pkl",
    shap_output_path: str = "shap_explainer.pkl",
    predictions_output_path: str = "predictions_with_scores.csv",
) -> dict[str, Any]:
    validate_runtime_dependencies()
    encoded = load_encoded_dataset(dataset_path, audit_dataset_path=audit_dataset_path)
    train_df, val_df, test_df = split_dataset(encoded)
    features = encoded.feature_columns
    x_train = train_df[features].copy()
    y_train = train_df["label_index"].to_numpy(dtype=int)
    x_val = val_df[features].copy()
    y_val = val_df["label_index"].to_numpy(dtype=int)
    x_test = test_df[features].copy()
    y_test = test_df["label_index"].to_numpy(dtype=int)
    system, tuning_payload = tune_models(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_columns=features,
    )
    shap_explainer = shap.TreeExplainer(system.xgb_model)
    scored_test = system.score_frame(x_test)
    test_metrics = evaluate_hybrid_predictions(y_test, scored_test)
    predictions_with_scores = build_predictions_with_scores(test_df, scored_test)
    predictions_with_scores.to_csv(predictions_output_path, index=False)
    sample_explanations = generate_sample_explanations(x_test, scored_test, system, shap_explainer)
    training_summary = {
        "dataset_path": dataset_path,
        "audit_dataset_path": audit_dataset_path if audit_dataset_path and Path(audit_dataset_path).exists() else None,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test_ood": int(len(test_df)),
        },
        "feature_count": int(len(features)),
        "feature_columns": features,
        "class_weights": CLASS_WEIGHTS,
        "forbidden_feature_check": {
            "passed": True,
            "forbidden_columns": sorted(RESTRICTED_FEATURE_COLUMNS),
        },
        "restricted_columns_dropped": sorted(RESTRICTED_FEATURE_COLUMNS | NON_NUMERIC_DROP_COLUMNS),
        "training_protocol": {
            "train_split": "train",
            "tune_split": "val",
            "evaluation_split": "test_ood",
        },
        "decision_logic": {
            "malicious_threshold": MALICIOUS_THRESHOLD,
            "suspicious_threshold": SUSPICIOUS_THRESHOLD,
            "low_risk_threshold": LOW_RISK_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "permission_request_threshold": PERMISSION_REQUEST_THRESHOLD,
        },
        "best_model_selection": tuning_payload,
        "test_ood_metrics": test_metrics,
        "top_global_feature_importance": system.global_feature_importance,
        "sample_prediction_explanations": sample_explanations,
        "output_files": {
            "trained_model": model_output_path,
            "feature_columns": feature_output_path,
            "shap_explainer": shap_output_path,
            "predictions_with_scores": predictions_output_path,
        },
    }
    joblib.dump(system, model_output_path)
    joblib.dump(features, feature_output_path)
    joblib.dump(shap_explainer, shap_output_path)
    with open("training_report.json", "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)
    return training_summary
