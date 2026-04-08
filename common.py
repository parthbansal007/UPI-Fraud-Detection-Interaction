from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


EXPECTED_EVENTS = {
    "click_link",
    "scroll",
    "type_text",
    "switch_app",
    "scan_qr",
    "download_file",
    "install_app",
    "grant_permission",
    "deny_permission",
    "start_screen_share",
    "stop_screen_share",
}

APP_EVENTS = {
    "open_app",
    "close_app",
    "switch_app",
}

ALLOWED_APPS = {
    "WhatsApp",
    "Browser",
    "YouTube",
    "Settings",
    "FileManager",
}

DEFAULT_REQUIRED_COLUMNS = [
    "session_id",
    "event_sequence",
    "label",
]

OPTIONAL_COLUMNS = [
    "timestamp_start",
    "timestamp_end",
    "event_timestamps",
    "user_type",
    "behavioral_features",
    "device_state",
    "generator_version",
    "scenario_family",
    "risk_score",
    "split",
    "split_reason",
    "upi_id",
    "payment_link",
    "qr_data",
]

LABEL_MAP = {
    "normal": "normal",
    "safe": "normal",
    "benign": "normal",
    "legit": "normal",
    "legitimate": "normal",
    "suspicious": "suspicious",
    "warn": "suspicious",
    "warning": "suspicious",
    "malicious": "malicious",
    "fraud": "malicious",
    "fraudulent": "malicious",
    "block": "malicious",
}


@dataclass
class ParsedRecord:
    events: list[str]
    event_timestamps: list[pd.Timestamp]
    gaps: list[float]
    session_duration: float
    behavioral_features: dict[str, Any]
    device_state: dict[str, Any]
    payment_link: str
    qr_data: str
    upi_id: str


def safe_literal_eval(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [] if value != value else {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return stripped
    return value


def parse_event_sequence(value: Any) -> list[str]:
    parsed = safe_literal_eval(value)
    if isinstance(parsed, str):
        if "|" in parsed:
            parsed = [part.strip() for part in parsed.split("|")]
        elif "," in parsed:
            parsed = [part.strip() for part in parsed.split(",")]
        else:
            parsed = [parsed.strip()]
    if not isinstance(parsed, list):
        raise ValueError("event_sequence could not be parsed into a list")
    events = [str(item).strip() for item in parsed if str(item).strip()]
    if not events:
        raise ValueError("event_sequence is empty")
    return events


def parse_timestamps(value: Any, expected_len: int) -> list[pd.Timestamp]:
    parsed = safe_literal_eval(value)
    if parsed in ({}, [], "", None):
        return []
    if isinstance(parsed, str):
        parsed = [part.strip() for part in parsed.split("|") if part.strip()]
    if not isinstance(parsed, list):
        return []
    timestamps = []
    for item in parsed:
        ts = pd.to_datetime(item, errors="coerce")
        if pd.isna(ts):
            return []
        timestamps.append(ts)
    if len(timestamps) != expected_len:
        return []
    return timestamps


def parse_mapping(value: Any) -> dict[str, Any]:
    parsed = safe_literal_eval(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def normalize_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing label")
    label = str(value).strip().lower()
    if label not in LABEL_MAP:
        raise ValueError(f"Unsupported label: {value}")
    return LABEL_MAP[label]


def extract_payment_link(row: pd.Series, behavioral_features: dict[str, Any]) -> str:
    for key in ["payment_link", "url", "link", "payment_url"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
        if key in behavioral_features and str(behavioral_features[key]).strip():
            return str(behavioral_features[key]).strip()
    return ""


def extract_upi_id(row: pd.Series, behavioral_features: dict[str, Any]) -> str:
    for key in ["upi_id", "payee_upi_id", "merchant_upi_id", "target_upi_id"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
        if key in behavioral_features and str(behavioral_features[key]).strip():
            return str(behavioral_features[key]).strip()
    link = extract_payment_link(row, behavioral_features)
    if link:
        parsed = urlparse(link)
        query = parse_qs(parsed.query)
        for key in ["pa", "upi_id", "payee"]:
            if key in query and query[key]:
                return query[key][0]
    return ""


def extract_qr_data(row: pd.Series, behavioral_features: dict[str, Any]) -> str:
    for key in ["qr_data", "qr_payload", "qr_string"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
        if key in behavioral_features and str(behavioral_features[key]).strip():
            return str(behavioral_features[key]).strip()
    return ""


def infer_start_end(row: pd.Series, timestamps: list[pd.Timestamp]) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    start = pd.to_datetime(row.get("timestamp_start"), errors="coerce")
    end = pd.to_datetime(row.get("timestamp_end"), errors="coerce")
    if pd.isna(start) and timestamps:
        start = timestamps[0]
    if pd.isna(end) and timestamps:
        end = timestamps[-1]
    return start, end


def parse_record(row: pd.Series) -> ParsedRecord:
    behavioral_features = parse_mapping(row.get("behavioral_features"))
    device_state = parse_mapping(row.get("device_state"))
    events = parse_event_sequence(row["event_sequence"])
    event_timestamps = parse_timestamps(row.get("event_timestamps"), len(events))
    start, end = infer_start_end(row, event_timestamps)
    if event_timestamps:
        gaps = np.diff(pd.Series(event_timestamps).astype("int64") / 1e9).tolist()
    else:
        duration_source = behavioral_features.get("session_duration") or row.get("session_duration")
        duration = float(duration_source) if duration_source not in [None, ""] else float(max(len(events) - 1, 1) * 5)
        gaps = [duration / max(len(events) - 1, 1)] * max(len(events) - 1, 0)
    if not pd.isna(start) and not pd.isna(end):
        session_duration = max((end - start).total_seconds(), 0.0)
    elif gaps:
        session_duration = float(sum(gaps))
    else:
        session_duration = float(max(len(events) - 1, 1) * 5)
    return ParsedRecord(
        events=events,
        event_timestamps=event_timestamps,
        gaps=[float(max(gap, 0.0)) for gap in gaps],
        session_duration=float(session_duration),
        behavioral_features=behavioral_features,
        device_state=device_state,
        payment_link=extract_payment_link(row, behavioral_features),
        qr_data=extract_qr_data(row, behavioral_features),
        upi_id=extract_upi_id(row, behavioral_features),
    )


def validate_event_vocabulary(events: list[str]) -> None:
    for event in events:
        if ":" in event:
            prefix, suffix = event.split(":", 1)
            if prefix not in APP_EVENTS or suffix not in ALLOWED_APPS:
                raise ValueError(f"Unsupported event token: {event}")
        elif event not in EXPECTED_EVENTS:
            raise ValueError(f"Unsupported event token: {event}")


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in DEFAULT_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    working = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in working.columns:
            working[col] = np.nan
    if "session_id" not in working or working["session_id"].isna().all():
        working["session_id"] = [f"session_{idx}" for idx in range(len(working))]
    working["label"] = working["label"].map(normalize_label)
    for idx, row in working.iterrows():
        parsed = parse_record(row)
        validate_event_vocabulary(parsed.events)
        if pd.isna(row.get("timestamp_start")) and parsed.event_timestamps:
            working.at[idx, "timestamp_start"] = parsed.event_timestamps[0]
        if pd.isna(row.get("timestamp_end")) and parsed.event_timestamps:
            working.at[idx, "timestamp_end"] = parsed.event_timestamps[-1]
        if pd.isna(row.get("upi_id")) and parsed.upi_id:
            working.at[idx, "upi_id"] = parsed.upi_id
        if pd.isna(row.get("payment_link")) and parsed.payment_link:
            working.at[idx, "payment_link"] = parsed.payment_link
        if pd.isna(row.get("qr_data")) and parsed.qr_data:
            working.at[idx, "qr_data"] = parsed.qr_data
    return working


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_schema(df)


def shannon_entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    _, counts = np.unique(items, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities + 1e-12)).sum())


def burstiness(gaps: list[float]) -> float:
    if not gaps:
        return 0.0
    values = np.asarray(gaps, dtype=float)
    mean = values.mean()
    std = values.std()
    denom = std + mean + 1e-9
    return float((std - mean) / denom)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def extract_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def contains_ip_address(url: str) -> int:
    domain = extract_domain(url)
    return int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?", domain)))


def is_https(url: str) -> int:
    if not url:
        return 0
    return int(urlparse(url).scheme.lower() == "https")


def count_redirect_hints(url: str) -> int:
    if not url:
        return 0
    lowered = url.lower()
    return sum(lowered.count(token) for token in ["redirect", "next=", "url=", "target=", "dest="])


def tokenize_upi_id(upi_id: str) -> tuple[str, str]:
    if "@" not in upi_id:
        return upi_id.lower(), ""
    name, handle = upi_id.split("@", 1)
    return name.lower(), handle.lower()


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def similarity_to_trusted(upi_id: str, trusted_handles: set[str]) -> float:
    name, handle = tokenize_upi_id(upi_id)
    if not upi_id:
        return 0.0
    if handle in trusted_handles:
        return 1.0
    comparisons = [levenshtein_distance(handle, trusted) for trusted in trusted_handles if trusted]
    if not comparisons:
        return 0.0
    best_distance = min(comparisons)
    scale = max(max(len(handle), 1), max(len(token) for token in trusted_handles))
    return float(max(0.0, 1.0 - best_distance / scale))
