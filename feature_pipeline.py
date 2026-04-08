from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from common import (
    APP_EVENTS,
    burstiness,
    contains_ip_address,
    count_redirect_hints,
    extract_domain,
    is_https,
    parse_record,
    safe_ratio,
    shannon_entropy,
    similarity_to_trusted,
)


def suspicious_subsequence_score(events: list[str]) -> float:
    tokens = " ".join(events)
    patterns = [
        "open_app:WhatsApp click_link open_app:Browser download_file install_app grant_permission",
        "scan_qr start_screen_share",
        "start_screen_share grant_permission",
    ]
    score = 0.0
    for pattern in patterns:
        if pattern in tokens:
            score += 1.0
    score += max(0, events.count("grant_permission") - 1) * 0.3
    return float(score)


@dataclass
class UPIFeaturePipeline:
    ngram_range: tuple[int, int] = (1, 3)
    max_ngram_features: int = 500
    top_markov_features: int = 60
    rapid_activity_threshold: float = 5.0
    trusted_handles: set[str] = field(default_factory=lambda: {"oksbi", "okicici", "okaxis", "ybl", "paytm", "ibl"})
    blacklist_tokens: set[str] = field(default_factory=lambda: {"verify", "secure", "reward", "refund", "support", "helpdesk"})
    domain_risk_lexicon: dict[str, float] = field(
        default_factory=lambda: {
            "bit.ly": 0.9,
            "tinyurl.com": 0.85,
            "cutt.ly": 0.8,
            "ngrok": 0.95,
            "vercel.app": 0.55,
            "appspot.com": 0.5,
        }
    )

    def __post_init__(self) -> None:
        self.ngram_vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=self.ngram_range,
            max_features=self.max_ngram_features,
        )
        self.stats_vectorizer = DictVectorizer(sparse=False)
        self.transition_features_: list[str] = []
        self.class_names_ = ["normal", "suspicious", "malicious"]

    def _sequence_text(self, events: list[str]) -> str:
        return " ".join(events)

    def _transition_counts(self, events: list[str]) -> dict[str, float]:
        counts: dict[str, float] = {}
        if len(events) < 2:
            return counts
        for left, right in zip(events[:-1], events[1:]):
            key = f"transition::{left}->{right}"
            counts[key] = counts.get(key, 0.0) + 1.0
        total = sum(counts.values()) or 1.0
        return {key: value / total for key, value in counts.items()}

    def _domain_reputation(self, url: str) -> float:
        domain = extract_domain(url)
        if not domain:
            return 0.5
        for token, risk in self.domain_risk_lexicon.items():
            if token in domain:
                return risk
        if domain.endswith(".gov.in") or domain.endswith(".bank"):
            return 0.1
        if domain.endswith(".in"):
            return 0.4
        return 0.5

    def _upi_features(self, upi_id: str) -> dict[str, float]:
        name, handle = ("", "")
        if upi_id:
            parts = upi_id.split("@", 1)
            name = parts[0].lower()
            handle = parts[1].lower() if len(parts) == 2 else ""
        digit_ratio = safe_ratio(sum(char.isdigit() for char in upi_id), len(upi_id))
        special_ratio = safe_ratio(sum(not char.isalnum() and char != "@" for char in upi_id), len(upi_id))
        blacklist_match = int(any(token in name for token in self.blacklist_tokens))
        return {
            "upi_length": float(len(upi_id)),
            "upi_digit_ratio": digit_ratio,
            "upi_special_ratio": special_ratio,
            "upi_missing_handle": float("@" not in upi_id),
            "upi_blacklist_match": float(blacklist_match),
            "upi_similarity_trusted": similarity_to_trusted(upi_id, self.trusted_handles),
            "upi_handle_known": float(handle in self.trusted_handles),
        }

    def _event_features(self, events: list[str], record) -> dict[str, float]:
        freq = {f"event_count::{event}": float(events.count(event)) for event in set(events)}
        app_event_count = 0
        app_switches = 0
        short_gaps = sum(gap <= 3.0 for gap in record.gaps)
        qr_events = events.count("scan_qr")
        link_events = events.count("click_link")
        permission_events = events.count("grant_permission") + events.count("deny_permission")
        for event in events:
            if any(event.startswith(f"{prefix}:") for prefix in APP_EVENTS):
                app_event_count += 1
            if event.startswith("switch_app"):
                app_switches += 1
        joined = " ".join(events)
        whatsapp_to_link = float(
            "open_app:WhatsApp click_link" in joined or "switch_app:WhatsApp click_link" in joined
        )
        avg_seconds_per_event = safe_ratio(record.session_duration, max(len(events), 1))
        suspicious_pattern_score = suspicious_subsequence_score(events)
        borderline_device_risk = float(
            (record.device_state.get("ip_risk_score", 0.0) or 0.0) >= 0.4 and (record.device_state.get("ip_risk_score", 0.0) or 0.0) <= 0.7
        )
        mixed_channel_flag = float(qr_events > 0 and link_events > 0)
        event_diversity_ratio = safe_ratio(len(set(events)), max(len(events), 1))
        borderline_interaction_score = link_events * 0.35 + qr_events * 0.3 + permission_events * 0.35
        borderline_behavior_score = (
            safe_ratio(link_events, len(events)) * 0.3
            + safe_ratio(permission_events, len(events)) * 0.3
            + safe_ratio(qr_events, len(events)) * 0.2
            + event_diversity_ratio * 0.2
        )
        return {
            **freq,
            "sequence_length": float(len(events)),
            "event_diversity": float(len(set(events))),
            "event_diversity_ratio": event_diversity_ratio,
            "sequence_entropy": shannon_entropy(events),
            "markov_unique_transitions": float(len(set(zip(events[:-1], events[1:])))) if len(events) > 1 else 0.0,
            "session_duration": record.session_duration,
            "avg_seconds_per_event": avg_seconds_per_event,
            "avg_gap": float(np.mean(record.gaps)) if record.gaps else 0.0,
            "std_gap": float(np.std(record.gaps)) if record.gaps else 0.0,
            "min_gap": float(np.min(record.gaps)) if record.gaps else 0.0,
            "max_gap": float(np.max(record.gaps)) if record.gaps else 0.0,
            "burstiness": burstiness(record.gaps),
            "click_rate": safe_ratio(events.count("click_link"), max(record.session_duration, 1.0)),
            "link_click_rate": safe_ratio(link_events, len(events)),
            "permission_ratio": safe_ratio(permission_events, len(events)),
            "permission_abuse_ratio": safe_ratio(permission_events, len(events)),
            "qr_usage_ratio": safe_ratio(qr_events, len(events)),
            "qr_to_action_ratio": safe_ratio(qr_events, len(events)),
            "rapid_activity_flag": float(avg_seconds_per_event < self.rapid_activity_threshold),
            "short_gap_ratio": safe_ratio(short_gaps, max(len(record.gaps), 1)),
            "link_qr_combo": float(qr_events > 0 and link_events > 0),
            "mixed_channel_flag": mixed_channel_flag,
            "screen_share_without_context": float("start_screen_share" in events and "open_app:WhatsApp" not in events),
            "grant_permission_abuse": safe_ratio(events.count("grant_permission"), max(permission_events, 1)),
            "suspicious_subsequence_score": suspicious_subsequence_score(events),
            "suspicious_pattern_score": suspicious_pattern_score,
            "borderline_device_risk": borderline_device_risk,
            "borderline_interaction_score": borderline_interaction_score,
            "behavioral_entropy_proxy": safe_ratio(len(set(events)), max(float(np.sqrt(max(len(events), 1))), 1.0)),
            "borderline_behavior_score": borderline_behavior_score,
            "whatsapp_link_pattern": whatsapp_to_link,
            "app_event_ratio": safe_ratio(app_event_count, len(events)),
            "switch_app_ratio": safe_ratio(app_switches, len(events)),
        }

    def _device_features(self, record) -> dict[str, float]:
        state = record.device_state
        ip_risk = state.get("ip_risk_score", 0.0)
        try:
            ip_risk = float(ip_risk)
        except Exception:
            ip_risk = 0.0
        return {
            "vpn": float(state.get("vpn", 0) or 0),
            "rooted": float(state.get("rooted", 0) or 0),
            "emulator": float(state.get("emulator", 0) or 0),
            "ip_risk_score": ip_risk,
            "device_anomaly_sum": float((state.get("vpn", 0) or 0) + (state.get("rooted", 0) or 0) + (state.get("emulator", 0) or 0)),
        }

    def _payment_features(self, record) -> dict[str, float]:
        url = record.payment_link or ""
        qr_data = record.qr_data or ""
        domain = extract_domain(url)
        contains_qr_upi = float("upi://" in qr_data.lower() or "pa=" in qr_data.lower())
        return {
            "payment_link_present": float(bool(url)),
            "payment_link_length": float(len(url)),
            "payment_domain_reputation": self._domain_reputation(url),
            "payment_has_ip_domain": float(contains_ip_address(url)),
            "payment_https": float(is_https(url)),
            "payment_redirect_hints": float(count_redirect_hints(url)),
            "payment_domain_length": float(len(domain)),
            "payment_domain_digits": float(sum(char.isdigit() for char in domain)),
            "qr_present": float(bool(qr_data)),
            "qr_length": float(len(qr_data)),
            "qr_contains_upi": contains_qr_upi,
        }

    def _behavioral_dict_features(self, record) -> dict[str, float]:
        features = {}
        for key, value in record.behavioral_features.items():
            if isinstance(value, bool):
                features[f"behavior::{key}"] = float(value)
            elif isinstance(value, (int, float)):
                features[f"behavior::{key}"] = float(value)
        return features

    def _build_feature_row(self, row: pd.Series) -> tuple[str, dict[str, float], dict[str, float]]:
        record = parse_record(row)
        events = record.events
        transition_counts = self._transition_counts(events)
        stats = {}
        stats.update(self._event_features(events, record))
        stats.update(self._device_features(record))
        stats.update(self._payment_features(record))
        stats.update(self._upi_features(record.upi_id))
        stats.update(self._behavioral_dict_features(record))
        stats["risk_score_input"] = float(row.get("risk_score")) if pd.notna(row.get("risk_score")) else 0.0
        return self._sequence_text(events), stats, transition_counts

    def fit(self, df: pd.DataFrame) -> "UPIFeaturePipeline":
        sequence_texts = []
        stat_dicts = []
        transition_totals: dict[str, float] = {}
        for _, row in df.iterrows():
            sequence_text, stats, transitions = self._build_feature_row(row)
            sequence_texts.append(sequence_text)
            stat_dicts.append(stats)
            for key, value in transitions.items():
                transition_totals[key] = transition_totals.get(key, 0.0) + value
        self.ngram_vectorizer.fit(sequence_texts)
        self.stats_vectorizer.fit(stat_dicts)
        sorted_transitions = sorted(transition_totals.items(), key=lambda item: item[1], reverse=True)
        self.transition_features_ = [key for key, _ in sorted_transitions[: self.top_markov_features]]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        sequence_texts = []
        stat_dicts = []
        transition_matrix = []
        for _, row in df.iterrows():
            sequence_text, stats, transitions = self._build_feature_row(row)
            sequence_texts.append(sequence_text)
            stat_dicts.append(stats)
            transition_matrix.append([transitions.get(key, 0.0) for key in self.transition_features_])
        ngram_features = self.ngram_vectorizer.transform(sequence_texts).toarray()
        stats_features = self.stats_vectorizer.transform(stat_dicts)
        transition_features = np.asarray(transition_matrix, dtype=float)
        return np.hstack([ngram_features, stats_features, transition_features])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        return (
            list(self.ngram_vectorizer.get_feature_names_out())
            + list(self.stats_vectorizer.get_feature_names_out())
            + self.transition_features_
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "UPIFeaturePipeline":
        return joblib.load(path)

    def explain_instance(self, row: pd.Series, top_k: int = 8) -> dict[str, Any]:
        _, stats, transitions = self._build_feature_row(row)
        transition_stats = {key: transitions.get(key, 0.0) for key in self.transition_features_ if transitions.get(key, 0.0) > 0}
        combined = {**stats, **transition_stats}
        ranked = sorted(combined.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        return {
            "top_features": [{"feature": key, "value": float(value)} for key, value in ranked],
            "suspicious_segments": self._extract_segments(parse_record(row).events),
        }

    def _extract_segments(self, events: list[str]) -> list[list[str]]:
        segments = []
        targets = [
            ["open_app:WhatsApp", "click_link", "open_app:Browser", "download_file", "install_app", "grant_permission"],
            ["scan_qr", "start_screen_share"],
            ["start_screen_share", "grant_permission"],
        ]
        for target in targets:
            target_len = len(target)
            for idx in range(0, max(len(events) - target_len + 1, 0)):
                window = events[idx : idx + target_len]
                if sum(left == right for left, right in zip(window, target)) >= max(2, target_len - 1):
                    segments.append(window)
        return segments

    def save_metadata(self, path: str) -> None:
        payload = {
            "class_names": self.class_names_,
            "feature_count": len(self.get_feature_names()),
            "top_markov_features": self.transition_features_,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
