from __future__ import annotations

from pathlib import Path
from typing import Any
from sklearn.preprocessing import StandardScaler
import pandas as pd

from fraud_system.hybrid_pipeline import HybridFraudDetector, PipelineConfig
_FEATURE_DETECTOR: HybridFraudDetector | None = None


def _get_feature_detector() -> HybridFraudDetector:
    global _FEATURE_DETECTOR
    if _FEATURE_DETECTOR is None:
        _FEATURE_DETECTOR = HybridFraudDetector(
            PipelineConfig(
                raw_data_path=Path("interaction_data/dataset_v2.csv"),
                encoded_data_path=Path("interaction_data/dataset_encoded_v2.csv"),
                artifact_dir=Path("models/interaction"),
            )
        )
    return _FEATURE_DETECTOR


def extract_url_features(url: str) -> dict[str, float]:
    detector = _get_feature_detector()
    return detector._extract_url_features(url or "https://unknown.local")


def extract_behavior_features(
    event_sequence: list[str] | None,
    event_timestamps: list[str] | None,
    session_duration: float | int | None = None,
) -> dict[str, float]:
    detector = _get_feature_detector()
    row = pd.Series(
        {
            "event_sequence_list": event_sequence or [],
            "event_timestamps_list": event_timestamps or [],
            "session_duration": float(session_duration) if session_duration is not None else 0.0,
            "behavioral_dict": {"session_duration": float(session_duration or 0.0)},
        }
    )
    return detector._extract_behavior_features(row)


def build_interaction_text(
    input_text: str,
    url: str,
    qr_data: str,
    device_info: dict[str, Any] | None = None,
) -> str:
    detector = _get_feature_detector()
    return detector._online_text(
        input_text=input_text,
        url=url,
        qr_data=qr_data,
        device_info=device_info or {},
    )


def extract_feature_bundle(
    input_text: str,
    url: str,
    qr_data: str = "",
    event_sequence: list[str] | None = None,
    event_timestamps: list[str] | None = None,
    session_duration: float | int | None = None,
    device_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "interaction_text": build_interaction_text(
            input_text=input_text,
            url=url,
            qr_data=qr_data,
            device_info=device_info,
        ),
        "url_features": extract_url_features(url=url),
        "behavior_features": extract_behavior_features(
            event_sequence=event_sequence,
            event_timestamps=event_timestamps,
            session_duration=session_duration,
        ),
    }
