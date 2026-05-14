from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fraud_system.hybrid_pipeline import HybridFraudDetector


@dataclass
class FinalizeConfig:
    model_dir: Path = Path("models/interaction")
    ensemble_report_path: Path = Path("models/interaction/ensemble_report.json")
    transformer_report_path: Path = Path("models/interaction/transformer/training_report.json")
    xgboost_report_path: Path = Path("models/interaction/xgboost_tuning_report.json")
    isolation_forest_report_path: Path = Path("models/interaction/isolation_forest_report.json")
    metadata_path: Path = Path("models/interaction/metadata.json")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_required_files(model_dir: Path) -> dict[str, str]:
    required = {
        "transformer_dir": model_dir / "transformer",
        "xgboost_model": model_dir / "xgboost_model.json",
        "isolation_forest_model": model_dir / "isolation_forest.joblib",
        "structured_preprocessor": model_dir / "structured_preprocessor.joblib",
        "metadata": model_dir / "metadata.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts for finalization: {missing}")
    return {name: str(path.resolve()) for name, path in required.items()}


def _local_transformer_identity(
    model_dir: Path,
    metadata: dict[str, Any],
    transformer_report: dict[str, Any],
) -> dict[str, Any]:
    config_path = model_dir / "transformer" / "config.json"
    local_config = _load_json(config_path) if config_path.exists() else {}
    model_type = str(local_config.get("model_type", "")).lower()
    architectures = [str(item) for item in local_config.get("architectures", [])]
    metadata_name = str(metadata.get("transformer_model_name", "")).strip()
    report_name = str(transformer_report.get("selected_backbone", "")).strip()

    inferred_name = metadata_name or report_name or "local-transformer"
    if model_type == "distilbert" and "distilbert" not in inferred_name.lower():
        inferred_name = "distilbert-base-uncased"
    elif model_type == "roberta" and "roberta" not in inferred_name.lower():
        inferred_name = "roberta-base"
    elif model_type == "deberta-v2" and "deberta" not in inferred_name.lower():
        inferred_name = "microsoft/deberta-v3-base"

    return {
        "local_model_name": inferred_name,
        "config_model_type": model_type,
        "config_architectures": architectures,
        "metadata_model_name": metadata_name,
        "report_selected_backbone": report_name,
        "report_matches_local": bool(not report_name or report_name == inferred_name),
    }


def _sync_transformer_report(
    report_path: Path,
    transformer_report: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    local_name = str(identity["local_model_name"])
    previous_name = str(transformer_report.get("selected_backbone", ""))
    if previous_name and previous_name != local_name:
        transformer_report["original_selected_backbone"] = previous_name
    transformer_report["selected_backbone"] = local_name
    transformer_report["artifact_consistency"] = {
        "status": "aligned_to_local_artifact",
        **identity,
    }
    transformer_report.setdefault("artifact_paths", {})
    transformer_report["artifact_paths"]["model_dir"] = str((report_path.parent).resolve())
    transformer_report["artifact_paths"]["training_report"] = str(report_path.resolve())
    report_path.write_text(json.dumps(transformer_report, indent=2), encoding="utf-8")


def _write_final_project_report(
    model_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    ensemble_report: dict[str, Any],
    runtime_report: dict[str, Any],
    calibration_report: dict[str, Any],
    transformer_report: dict[str, Any],
    xgboost_report: dict[str, Any],
    isolation_report: dict[str, Any],
    smoke: dict[str, Any],
) -> Path:
    final_path = model_dir / "final_project_report.json"
    payload = {
        "status": "ready_for_inference",
        "model_dir": str(model_dir.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "class_order": metadata.get("class_order", ["normal", "suspicious", "malicious"]),
        "input_modes": {
            "single": "input_text + url + optional qr_data + optional device_info_json",
            "batch": "CSV with online fields, or raw interaction columns such as event_sequence, event_timestamps, device_state, and behavioral_features",
        },
        "output_columns": [
            "session_id",
            "interaction_label",
            "interaction_risk_score",
            "interaction_risk_level",
            "normal_probability",
            "suspicious_probability",
            "malicious_probability",
        ],
        "selected_runtime": {
            "transformer_model_name": metadata.get("transformer_model_name"),
            "max_length": metadata.get("max_length"),
            "ensemble_weights": metadata.get("ensemble_weights"),
            "malicious_threshold": metadata.get("malicious_threshold"),
            "suspicious_threshold": metadata.get("suspicious_threshold"),
        },
        "quality_summary": {
            "calibrated_runtime_test_ood": calibration_report.get("selected", {}).get("test_ood", {}),
            "historical_training_test_ood": runtime_report.get("test_ood_metrics", {}),
            "ensemble_test_ood": ensemble_report.get("split_metrics", {}).get("test_ood", {}),
            "xgboost_test_ood": xgboost_report.get("test_ood_metrics", {}),
            "transformer_test_ood": transformer_report.get("test_ood_metrics", {}),
            "isolation_forest": isolation_report.get("normalization", {}),
        },
        "inference_smoke_test": {
            "predicted_label": smoke.get("predicted_label"),
            "risk_level": smoke.get("risk_level"),
            "fraud_probability": smoke.get("fraud_probability"),
        },
        "recommended_commands": {
            "single_inference": "python main.py infer --input-text \"Urgent UPI verification needed now\" --url \"http://secure-kyc-update.in/verify-now?ref=upi\"",
            "batch_inference": "python main.py infer --input-file interaction_data/dataset_v2.csv --output-path outputs/predictions.csv",
            "evaluate": "python main.py evaluate --split test_ood",
        },
    }
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return final_path


def finalize_trained_fraud_models(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = FinalizeConfig(**(config or {}))
    cfg.model_dir = Path(cfg.model_dir)
    cfg.ensemble_report_path = Path(cfg.ensemble_report_path)
    cfg.transformer_report_path = Path(cfg.transformer_report_path)
    cfg.xgboost_report_path = Path(cfg.xgboost_report_path)
    cfg.isolation_forest_report_path = Path(cfg.isolation_forest_report_path)
    cfg.metadata_path = Path(cfg.metadata_path)

    artifact_paths = _assert_required_files(cfg.model_dir)
    metadata = _load_json(cfg.metadata_path)
    ensemble_report = _load_json(cfg.ensemble_report_path)
    transformer_report = _load_json(cfg.transformer_report_path)
    xgboost_report = _load_json(cfg.xgboost_report_path)
    isolation_report = _load_json(cfg.isolation_forest_report_path)
    runtime_report_path = cfg.model_dir / "evaluation_report.json"
    runtime_report = _load_json(runtime_report_path) if runtime_report_path.exists() else {}
    calibration_report_path = cfg.model_dir / "runtime_calibration_report.json"
    calibration_report = _load_json(calibration_report_path) if calibration_report_path.exists() else {}
    transformer_identity = _local_transformer_identity(
        model_dir=cfg.model_dir,
        metadata=metadata,
        transformer_report=transformer_report,
    )
    _sync_transformer_report(
        report_path=cfg.transformer_report_path,
        transformer_report=transformer_report,
        identity=transformer_identity,
    )

    tuned_weights = ensemble_report.get("tuned_weights", {})
    tuned_thresholds = ensemble_report.get("tuned_thresholds", {})
    if not {"w_transformer", "w_xgboost", "w_anomaly"}.issubset(tuned_weights):
        raise ValueError("Ensemble report missing tuned weights.")
    if not {"malicious_threshold", "suspicious_threshold"}.issubset(tuned_thresholds):
        raise ValueError("Ensemble report missing tuned thresholds.")

    selected_weights = [
        float(tuned_weights["w_transformer"]),
        float(tuned_weights["w_xgboost"]),
        float(tuned_weights["w_anomaly"]),
    ]
    selected_thresholds = {
        "malicious_threshold": float(tuned_thresholds["malicious_threshold"]),
        "suspicious_threshold": float(tuned_thresholds["suspicious_threshold"]),
    }
    selection_source = "ensemble_report"

    runtime_weights = runtime_report.get("config", {}).get("ensemble_weights")
    runtime_thresholds = runtime_report.get("thresholds", {})
    if (
        isinstance(runtime_weights, list)
        and len(runtime_weights) == 3
        and {"malicious_threshold", "suspicious_threshold"}.issubset(runtime_thresholds)
    ):
        selected_weights = [float(value) for value in runtime_weights]
        selected_thresholds = {
            "malicious_threshold": float(runtime_thresholds["malicious_threshold"]),
            "suspicious_threshold": float(runtime_thresholds["suspicious_threshold"]),
        }
        selection_source = "runtime_evaluation_report"

    calibration_selected = calibration_report.get("selected", {})
    calibration_weights = calibration_selected.get("weights")
    if (
        isinstance(calibration_weights, list)
        and len(calibration_weights) == 3
        and "malicious_threshold" in calibration_selected
        and "suspicious_threshold" in calibration_selected
    ):
        selected_weights = [float(value) for value in calibration_weights]
        selected_thresholds = {
            "malicious_threshold": float(calibration_selected["malicious_threshold"]),
            "suspicious_threshold": float(calibration_selected["suspicious_threshold"]),
        }
        selection_source = "runtime_calibration_report"

    metadata["ensemble_weights"] = selected_weights
    metadata["malicious_threshold"] = selected_thresholds["malicious_threshold"]
    metadata["suspicious_threshold"] = selected_thresholds["suspicious_threshold"]

    metadata["transformer_model_name"] = str(transformer_identity["local_model_name"])
    max_length = transformer_report.get("config", {}).get("max_length")
    if max_length is not None:
        metadata["max_length"] = int(max_length)

    metadata["finalized"] = {
        "status": "ready_for_inference",
        "transformer_artifact_consistency": transformer_identity,
        "reports": {
            "ensemble_report": str(cfg.ensemble_report_path.resolve()),
            "transformer_report": str(cfg.transformer_report_path.resolve()),
            "xgboost_report": str(cfg.xgboost_report_path.resolve()),
            "isolation_forest_report": str(cfg.isolation_forest_report_path.resolve()),
            "runtime_evaluation_report": str(runtime_report_path.resolve()) if runtime_report_path.exists() else None,
            "runtime_calibration_report": str(calibration_report_path.resolve()) if calibration_report_path.exists() else None,
        },
        "selected": {
            "ensemble_weights": metadata["ensemble_weights"],
            "malicious_threshold": metadata["malicious_threshold"],
            "suspicious_threshold": metadata["suspicious_threshold"],
            "selection_source": selection_source,
        },
        "quality_summary": {
            "calibrated_runtime_test_ood_metrics": calibration_report.get("selected", {}).get("test_ood", {}),
            "historical_training_test_ood_metrics": runtime_report.get("test_ood_metrics", {}),
            "ensemble_val_metrics": ensemble_report.get("split_metrics", {}).get("val", {}),
            "ensemble_test_ood_metrics": ensemble_report.get("split_metrics", {}).get("test_ood", {}),
            "xgboost_test_ood_metrics": xgboost_report.get("test_ood_metrics", {}),
            "isolation_forest_normalization": isolation_report.get("normalization", {}),
        },
    }

    cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Smoke test: ensure loader and runtime inference remain compatible.
    detector = HybridFraudDetector.load(cfg.model_dir)
    smoke = detector.detect_fraud(
        input_text="Urgent KYC verification required for UPI account update.",
        url="https://secure-kyc-update.in/verify-now",
        qr_data="",
        device_info={"vpn": 1, "rooted": 0, "emulator": 0, "ip_risk_score": 0.72},
    )
    final_report_path = _write_final_project_report(
        model_dir=cfg.model_dir,
        metadata_path=cfg.metadata_path,
        metadata=metadata,
        ensemble_report=ensemble_report,
        runtime_report=runtime_report,
        calibration_report=calibration_report,
        transformer_report=transformer_report,
        xgboost_report=xgboost_report,
        isolation_report=isolation_report,
        smoke=smoke,
    )

    return {
        "status": "finalized",
        "model_dir": str(cfg.model_dir.resolve()),
        "saved_metadata": str(cfg.metadata_path.resolve()),
        "final_project_report": str(final_report_path.resolve()),
        "artifact_paths": artifact_paths,
        "selected_ensemble_weights": metadata["ensemble_weights"],
        "selected_thresholds": {
            "malicious_threshold": metadata["malicious_threshold"],
            "suspicious_threshold": metadata["suspicious_threshold"],
        },
        "selection_source": selection_source,
        "inference_smoke_test": {
            "predicted_label": smoke["predicted_label"],
            "fraud_probability": float(smoke["fraud_probability"]),
            "risk_level": smoke["risk_level"],
        },
        "config": {
            **asdict(cfg),
            "model_dir": str(cfg.model_dir),
            "ensemble_report_path": str(cfg.ensemble_report_path),
            "transformer_report_path": str(cfg.transformer_report_path),
            "xgboost_report_path": str(cfg.xgboost_report_path),
            "isolation_forest_report_path": str(cfg.isolation_forest_report_path),
            "metadata_path": str(cfg.metadata_path),
        },
    }
