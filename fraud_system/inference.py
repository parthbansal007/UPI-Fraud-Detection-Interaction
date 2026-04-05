from __future__ import annotations

from pathlib import Path
from typing import Any

from .hybrid_pipeline import HybridFraudDetector

_DETECTOR_CACHE: dict[str, HybridFraudDetector] = {}


def _resolve_model_dir(artifact_dir: str | Path) -> Path:
    requested = Path(artifact_dir)
    if (requested / "metadata.json").exists():
        return requested
    legacy = Path("artifacts")
    if (legacy / "metadata.json").exists():
        return legacy
    return requested


def _load_detector(artifact_dir: str | Path = "models/interaction") -> HybridFraudDetector:
    resolved = _resolve_model_dir(artifact_dir)
    key = str(resolved.resolve())
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = HybridFraudDetector.load(resolved)
    return _DETECTOR_CACHE[key]


def detect_fraud(
    input_text: str,
    url: str,
    qr_data: str,
    device_info: dict[str, Any] | None = None,
    artifact_dir: str | Path = "models/interaction",
) -> dict[str, Any]:
    detector = _load_detector(artifact_dir)
    return detector.detect_fraud(input_text=input_text, url=url, qr_data=qr_data, device_info=device_info or {})
