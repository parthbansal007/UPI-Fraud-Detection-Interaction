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

    tuned_weights = ensemble_report.get("tuned_weights", {})
    tuned_thresholds = ensemble_report.get("tuned_thresholds", {})
    if not {"w_transformer", "w_xgboost", "w_anomaly"}.issubset(tuned_weights):
        raise ValueError("Ensemble report missing tuned weights.")
    if not {"malicious_threshold", "suspicious_threshold"}.issubset(tuned_thresholds):
        raise ValueError("Ensemble report missing tuned thresholds.")

    metadata["ensemble_weights"] = [
        float(tuned_weights["w_transformer"]),
        float(tuned_weights["w_xgboost"]),
        float(tuned_weights["w_anomaly"]),
    ]
    metadata["malicious_threshold"] = float(tuned_thresholds["malicious_threshold"])
    metadata["suspicious_threshold"] = float(tuned_thresholds["suspicious_threshold"])

    selected_backbone = transformer_report.get("selected_backbone")
    if selected_backbone:
        metadata["transformer_model_name"] = str(selected_backbone)
    max_length = transformer_report.get("config", {}).get("max_length")
    if max_length is not None:
        metadata["max_length"] = int(max_length)

    metadata["finalized"] = {
        "status": "ready_for_inference",
        "reports": {
            "ensemble_report": str(cfg.ensemble_report_path.resolve()),
            "transformer_report": str(cfg.transformer_report_path.resolve()),
            "xgboost_report": str(cfg.xgboost_report_path.resolve()),
            "isolation_forest_report": str(cfg.isolation_forest_report_path.resolve()),
        },
        "selected": {
            "ensemble_weights": metadata["ensemble_weights"],
            "malicious_threshold": metadata["malicious_threshold"],
            "suspicious_threshold": metadata["suspicious_threshold"],
        },
        "quality_summary": {
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

    return {
        "status": "finalized",
        "model_dir": str(cfg.model_dir.resolve()),
        "saved_metadata": str(cfg.metadata_path.resolve()),
        "artifact_paths": artifact_paths,
        "selected_ensemble_weights": metadata["ensemble_weights"],
        "selected_thresholds": {
            "malicious_threshold": metadata["malicious_threshold"],
            "suspicious_threshold": metadata["suspicious_threshold"],
        },
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
