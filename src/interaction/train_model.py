from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_system.hybrid_pipeline import HybridFraudDetector, PipelineConfig, train_hybrid_detector

from .preprocess import prepare_interaction_data, resolve_interaction_paths

DEFAULT_DATA_DIR = Path("interaction_data")
DEFAULT_MODEL_DIR = Path("models/interaction")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LEGACY_ARTIFACT_DIR = Path("artifacts")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_pipeline_config(config: dict[str, Any] | None = None) -> tuple[PipelineConfig, dict[str, Path]]:
    cfg = dict(config or {})
    data_dir = Path(cfg.get("data_dir", DEFAULT_DATA_DIR))
    resolved = resolve_interaction_paths(data_dir=data_dir)

    raw_data_path = Path(cfg.get("raw_data_path", resolved["raw_data_path"]))
    encoded_data_path = Path(cfg.get("encoded_data_path", resolved["encoded_data_path"]))
    model_dir = Path(cfg.get("model_dir", DEFAULT_MODEL_DIR))

    pipeline_cfg = PipelineConfig(
        seed=int(cfg.get("seed", 42)),
        raw_data_path=raw_data_path,
        encoded_data_path=encoded_data_path,
        artifact_dir=model_dir,
        max_length=int(cfg.get("max_length", 96)),
        transformer_epochs=float(cfg.get("epochs", 4.0)),
        train_batch_size=int(cfg.get("train_batch_size", 16)),
        eval_batch_size=int(cfg.get("eval_batch_size", 32)),
        quick_mode=bool(cfg.get("quick", False)),
    )
    if pipeline_cfg.quick_mode:
        pipeline_cfg.transformer_learning_rates = (3e-5,)
        pipeline_cfg.xgb_param_grid = [pipeline_cfg.xgb_param_grid[0]]
        pipeline_cfg.xgb_estimators = int(cfg.get("quick_estimators", 300))
        pipeline_cfg.latency_runs = int(cfg.get("latency_runs", 30))

    return pipeline_cfg, {"data_dir": data_dir, "model_dir": model_dir, "output_dir": Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))}


def migrate_legacy_artifacts(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    legacy_dir: str | Path = DEFAULT_LEGACY_ARTIFACT_DIR,
) -> bool:
    model_path = Path(model_dir)
    legacy_path = Path(legacy_dir)
    if (model_path / "metadata.json").exists():
        return False
    if not (legacy_path / "metadata.json").exists():
        return False
    model_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(legacy_path, model_path, dirs_exist_ok=True)
    return True


def _publish_team_compatibility_files(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, Path]:
    model_path = Path(model_dir)
    parent_models_dir = model_path.parent
    parent_models_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridFraudDetector.load(model_path)
    xgb_pkl = parent_models_dir / "xgboost_model.pkl"
    scaler_pkl = parent_models_dir / "scaler.pkl"
    joblib.dump(detector.xgb_model, xgb_pkl)
    joblib.dump(detector.preprocessor, scaler_pkl)
    return {"xgboost_model_pkl": xgb_pkl, "scaler_pkl": scaler_pkl}


def _write_metrics_from_report(report: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "accuracy": report.get("test_ood_metrics", {}).get("accuracy"),
        "precision_by_class": report.get("test_ood_metrics", {}).get("precision_by_class", {}),
        "recall_by_class": report.get("test_ood_metrics", {}).get("recall_by_class", {}),
        "f1_by_class": report.get("test_ood_metrics", {}).get("f1_by_class", {}),
        "malicious_precision": report.get("test_ood_metrics", {}).get("malicious_precision"),
        "malicious_recall": report.get("test_ood_metrics", {}).get("malicious_recall"),
        "validation": report.get("validation_metrics", {}),
        "latency": report.get("latency", {}),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_path


def _publish_default_predictions(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path | None:
    example_path = Path(model_dir) / "example_predictions.csv"
    if not example_path.exists():
        return None
    df = pd.read_csv(example_path)
    standardized = pd.DataFrame(
        {
            "session_id": df.get("session_id"),
            "interaction_label": df.get("predicted_label"),
            "interaction_risk_score": df.get("fraud_probability"),
            "interaction_risk_level": df.get("risk_level"),
            "true_label": df.get("true_label"),
        }
    )
    out_path = Path(output_dir) / "predictions.csv"
    _ensure_parent(out_path)
    standardized.to_csv(out_path, index=False)
    return out_path


def load_interaction_model(model_dir: str | Path = DEFAULT_MODEL_DIR) -> HybridFraudDetector:
    model_path = Path(model_dir)
    if not (model_path / "metadata.json").exists():
        migrate_legacy_artifacts(model_dir=model_path)
    if not (model_path / "metadata.json").exists():
        raise FileNotFoundError(f"Interaction model not found in: {model_path}")
    return HybridFraudDetector.load(model_path)


def detect_fraud(
    input_text: str,
    url: str,
    qr_data: str,
    device_info: dict[str, Any] | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    detector = load_interaction_model(model_dir=model_dir)
    return detector.detect_fraud(
        input_text=input_text,
        url=url,
        qr_data=qr_data,
        device_info=device_info or {},
    )


def _extract_text(record: dict[str, Any]) -> str:
    candidates = ["input_text", "text", "message", "text_input", "sms_text", "whatsapp_text"]
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _extract_url(record: dict[str, Any]) -> str:
    candidates = ["url", "payment_url", "upi_url", "url_input", "link"]
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "https://unknown.local"


def _extract_qr(record: dict[str, Any]) -> str:
    candidates = ["qr_data", "qr_payload", "qr", "qr_code"]
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _extract_device_info(record: dict[str, Any]) -> dict[str, Any]:
    base = dict(record.get("device_info", {}) if isinstance(record.get("device_info"), dict) else {})
    passthrough_keys = [
        "vpn",
        "rooted",
        "emulator",
        "ip_risk_score",
        "risk_score",
        "scenario_family",
        "user_type",
        "num_events",
        "num_links_clicked",
        "num_qr_scans",
        "num_permission_requests",
        "session_duration",
    ]
    for key in passthrough_keys:
        if key in record and record[key] is not None:
            base[key] = record[key]
    return base


def predict_interaction_risk(
    df_or_inputs: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    if isinstance(df_or_inputs, pd.DataFrame):
        records = df_or_inputs.to_dict(orient="records")
    elif isinstance(df_or_inputs, dict):
        records = [df_or_inputs]
    elif isinstance(df_or_inputs, list):
        records = [dict(item) for item in df_or_inputs]
    else:
        raise TypeError("df_or_inputs must be a DataFrame, dict, or list[dict].")

    detector = load_interaction_model(model_dir=model_dir)
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        session_id = str(record.get("session_id", f"interaction_{idx:06d}"))
        result = detector.detect_fraud(
            input_text=_extract_text(record),
            url=_extract_url(record),
            qr_data=_extract_qr(record),
            device_info=_extract_device_info(record),
        )
        rows.append(
            {
                "session_id": session_id,
                "interaction_label": result["predicted_label"],
                "interaction_risk_score": float(result["fraud_probability"]),
                "interaction_risk_level": result["risk_level"],
                "normal_probability": float(result["class_probabilities"]["normal"]),
                "suspicious_probability": float(result["class_probabilities"]["suspicious"]),
                "malicious_probability": float(result["class_probabilities"]["malicious"]),
            }
        )
    return pd.DataFrame(rows)


def train_interaction_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline_cfg, resolved = _build_pipeline_config(config=config)
    prepare_interaction_data(data_dir=resolved["data_dir"])
    report = train_hybrid_detector(config=pipeline_cfg)
    _publish_team_compatibility_files(model_dir=resolved["model_dir"])
    _write_metrics_from_report(report=report, output_dir=resolved["output_dir"])
    _publish_default_predictions(model_dir=resolved["model_dir"], output_dir=resolved["output_dir"])
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train interaction model and publish team artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = train_interaction_model(
        {
            "data_dir": args.data_dir,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "quick": args.quick,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

