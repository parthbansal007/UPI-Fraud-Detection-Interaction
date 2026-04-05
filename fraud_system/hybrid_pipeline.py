from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

LABEL_ORDER = ["normal", "suspicious", "malicious"]
URL_KEYWORDS = [
    "upi",
    "pay",
    "collect",
    "verify",
    "kyc",
    "refund",
    "reward",
    "claim",
    "otp",
    "bank",
    "urgent",
    "secure",
    "wallet",
]
BLOCKED_MODEL_FEATURE_COLUMNS = frozenset(
    {
        "risk_score",
        "scenario_family",
        "label_id",
        "split",
        "split_reason",
        "generator_version",
        "session_id",
    }
)
BLOCKED_ENCODED_COLUMNS = BLOCKED_MODEL_FEATURE_COLUMNS - {"session_id"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:12], 16)


def _safe_json_load(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _find_column(columns: list[str], include_terms: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for key, original in lowered.items():
        if any(term in key for term in include_terms):
            return original
    return None


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(ch) / len(text) for ch in set(text)]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Keep CE loss in fp32 to avoid dtype mismatches under mixed precision.
        weights = self.class_weights.to(device=logits.device, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits.float(), labels)
        if return_outputs:
            return loss, outputs
        return loss


class HFClassifierWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


@dataclass
class PipelineConfig:
    seed: int = 42
    raw_data_path: Path = Path("interaction_data/dataset_v2.csv")
    encoded_data_path: Path = Path("interaction_data/dataset_encoded_v2.csv")
    artifact_dir: Path = Path("models/interaction")
    transformer_model_name: str = "microsoft/deberta-v3-base"
    fallback_transformer_model_name: str = "distilbert-base-uncased"
    max_length: int = 96
    train_batch_size: int = 16
    eval_batch_size: int = 32
    transformer_epochs: float = 5.0
    transformer_learning_rates: tuple[float, ...] = (2e-5, 3e-5)
    transformer_weight_decay: float = 0.01
    transformer_patience: int = 2
    xgb_param_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"max_depth": 4, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
            {"max_depth": 5, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.85},
            {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8},
        ]
    )
    xgb_estimators: int = 700
    xgb_early_stopping_rounds: int = 50
    iso_estimators: int = 300
    iso_contamination: float = 0.10
    min_malicious_recall: float = 0.30
    ensemble_weights: tuple[float, float, float] = (0.35, 0.55, 0.10)
    onnx_opset: int = 17
    onnx_vendor_paths: tuple[str, ...] = ("C:/onnxlib",)
    latency_runs: int = 100
    quick_mode: bool = False

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


class HybridFraudDetector:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.config.ensure_dirs()
        _set_seed(self.config.seed)

        self.label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        self.tokenizer: AutoTokenizer | None = None
        self.transformer_model: AutoModelForSequenceClassification | None = None
        self.transformer_model_name: str | None = None
        self.preprocessor: ColumnTransformer | None = None
        self.xgb_model: xgb.XGBClassifier | None = None
        self.iforest: IsolationForest | None = None

        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.structured_cols: list[str] = []
        self.structured_feature_names_out: list[str] = []
        self.full_feature_names: list[str] = []
        self.numeric_defaults: dict[str, float] = {}
        self.categorical_defaults: dict[str, str] = {}

        self._emb_mean: np.ndarray | None = None
        self._emb_std: np.ndarray | None = None

        self.anomaly_scale_low: float = 0.0
        self.anomaly_scale_high: float = 1.0
        self.malicious_threshold: float = 0.80
        self.suspicious_threshold: float = 0.55

    def _load_data(self) -> pd.DataFrame:
        raw_df = pd.read_csv(self.config.raw_data_path)
        encoded_df = pd.read_csv(self.config.encoded_data_path)

        encoded_drop = [col for col in BLOCKED_ENCODED_COLUMNS if col in encoded_df.columns]
        encoded_trim = encoded_df.drop(columns=encoded_drop, errors="ignore")
        merged = raw_df.merge(encoded_trim, on="session_id", how="left", suffixes=("", "_enc"))

        merged["event_sequence_list"] = merged["event_sequence"].apply(lambda x: _safe_json_load(x, []))
        merged["event_timestamps_list"] = merged["event_timestamps"].apply(lambda x: _safe_json_load(x, []))
        merged["device_state_dict"] = merged["device_state"].apply(lambda x: _safe_json_load(x, {}))
        merged["behavioral_dict"] = merged["behavioral_features"].apply(lambda x: _safe_json_load(x, {}))

        return merged

    def _synthesize_url(self, row: pd.Series) -> str:
        events = row.get("event_sequence_list", [])
        if not isinstance(events, list):
            events = []
        event_tokens = [str(evt).lower() for evt in events]
        links = int(_to_float(row.get("num_links_clicked", row.get("behavioral_dict", {}).get("num_links_clicked", 0))))
        qr_scans = int(_to_float(row.get("num_qr_scans", row.get("behavioral_dict", {}).get("num_qr_scans", 0))))
        permissions = int(
            _to_float(row.get("num_permission_requests", row.get("behavioral_dict", {}).get("num_permission_requests", 0)))
        )
        has_risky_event = any(
            ("click_link" in token) or ("scan_qr" in token) or ("permission" in token)
            for token in event_tokens
        )

        suspicious = has_risky_event or links > 0 or qr_scans > 0 or permissions > 0
        behavior_signature = "|".join(event_tokens[:16]) or "no_events"
        profile = f"l{links}|q{qr_scans}|p{permissions}|e{len(event_tokens)}|sig:{behavior_signature}"

        benign_domains = [
            "www.payments-bank.in",
            "offers.trusted-merchant.in",
            "support.upi-safe.in",
            "chat.family-updates.in",
            "portal.utility-bills.in",
        ]
        risky_domains = [
            "upi-verify-security.in",
            "wallet-refund-fast.in",
            "secure-kyc-update.in",
            "support-upi-alert.in",
            "reward-claim-now.in",
            "paytm-upgrade-center.in",
        ]
        domain_pool = risky_domains if suspicious else benign_domains
        domain = domain_pool[_stable_int(profile) % len(domain_pool)]

        suspicious_paths = [
            "kyc/verify-now",
            "payment/collect/pending",
            "wallet/refund/urgent",
            "upi/upgrade/security-check",
            "bank/otp/confirm",
        ]
        benign_paths = [
            "offers/cashback",
            "help/faq",
            "merchant/status",
            "transactions/history",
            "notifications/update",
        ]
        path_pool = suspicious_paths if suspicious else benign_paths
        path = path_pool[_stable_int(f"path|{profile}") % len(path_pool)]

        use_http = suspicious and (_stable_int(f"http|{profile}") % 4 == 0)
        scheme = "http" if use_http else "https"
        return f"{scheme}://{domain}/{path}?src=interaction"

    def _synthesize_qr_data(self, row: pd.Series) -> str:
        qr_scans = int(_to_float(row.get("num_qr_scans", row.get("behavioral_dict", {}).get("num_qr_scans", 0))))
        if qr_scans <= 0:
            return ""

        num_events = int(_to_float(row.get("num_events", row.get("behavioral_dict", {}).get("num_events", 0))))
        num_links = int(_to_float(row.get("num_links_clicked", row.get("behavioral_dict", {}).get("num_links_clicked", 0))))
        num_permissions = int(
            _to_float(row.get("num_permission_requests", row.get("behavioral_dict", {}).get("num_permission_requests", 0)))
        )
        user_type = str(row.get("user_type", "unknown"))
        qr_profile = f"q{qr_scans}|e{num_events}|l{num_links}|p{num_permissions}|u{user_type}"
        stable_id = _stable_int(qr_profile)
        amount = 199 + (stable_id % 1800)
        pa = f"merchant{stable_id % 10000}@upi"
        return f"upi://pay?pa={pa}&pn=Merchant&am={amount}&cu=INR"

    def _synthesize_message(self, row: pd.Series, url: str, qr_data: str) -> str:
        events = row.get("event_sequence_list", [])
        if not isinstance(events, list):
            events = []
        event_text = " ".join(str(evt).replace(":", " ") for evt in events[:16])
        event_tokens = [str(evt).lower() for evt in events]

        user_type = str(row.get("user_type", "user"))
        has_link_or_permission = any(("click_link" in token) or ("permission" in token) for token in event_tokens)
        has_qr = any("scan_qr" in token for token in event_tokens)

        if has_link_or_permission:
            prefix = "WhatsApp alert: UPI profile verification needed before payment release."
        elif has_qr:
            prefix = "SMS: Scan the provided QR to complete pending wallet settlement."
        else:
            prefix = "SMS update: Payment context review requested for account activity."

        qr_part = f" QR payload: {qr_data}." if qr_data else ""
        return (
            f"{prefix} User profile {user_type}. "
            f"Observed sequence: {event_text}. Link: {url}.{qr_part}"
        )

    def _extract_url_features(self, url: str) -> dict[str, float]:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path or ""
        query = parsed.query or ""
        full = url.lower()

        domain_no_port = domain.split(":")[0]
        is_ip = 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain_no_port or "") else 0
        domain_age_days = 30 + (_stable_int(domain_no_port or "unknown-domain") % 3650)

        keyword_flags = {f"url_kw_{kw}": float(kw in full) for kw in URL_KEYWORDS}
        features: dict[str, float] = {
            "url_length": float(len(url)),
            "url_entropy": _shannon_entropy(url),
            "url_domain_length": float(len(domain)),
            "url_path_length": float(len(path)),
            "url_query_length": float(len(query)),
            "url_num_digits": float(sum(ch.isdigit() for ch in url)),
            "url_num_special_chars": float(sum(not ch.isalnum() for ch in url)),
            "url_has_https": float(parsed.scheme.lower() == "https"),
            "url_has_ip": float(is_ip),
            "url_subdomain_count": float(max(domain_no_port.count(".") - 1, 0)),
            "url_domain_age_days_mock": float(domain_age_days),
            "url_keyword_count": float(sum(keyword_flags.values())),
        }
        features.update(keyword_flags)
        return features

    def _extract_behavior_features(self, row: pd.Series) -> dict[str, float]:
        events = row.get("event_sequence_list", [])
        if not isinstance(events, list):
            events = []
        timestamps = row.get("event_timestamps_list", [])
        if not isinstance(timestamps, list):
            timestamps = []

        seq_len = float(len(events))
        gap_values: list[float] = []
        if len(timestamps) > 1:
            ts = pd.to_datetime(pd.Series(timestamps), errors="coerce", format="ISO8601")
            diffs = ts.diff().dt.total_seconds().dropna()
            gap_values = [float(v) for v in diffs.tolist() if pd.notna(v)]

        if not gap_values:
            session_duration = _to_float(row.get("session_duration", row.get("behavioral_dict", {}).get("session_duration", 0)))
            approx_gap = session_duration / max(seq_len - 1.0, 1.0)
            gap_values = [approx_gap]

        event_tokens = [str(e).lower() for e in events]
        switch_count = sum(1 for e in event_tokens if "switch_app" in e)
        click_count = sum(1 for e in event_tokens if "click_link" in e)
        qr_count = sum(1 for e in event_tokens if "scan_qr" in e)
        type_count = sum(1 for e in event_tokens if "type_text" in e)
        perm_count = sum(1 for e in event_tokens if "permission" in e)
        open_apps = [e.split(":", 1)[1] for e in event_tokens if e.startswith("open_app:") and ":" in e]

        return {
            "sequence_length": seq_len,
            "event_gap_mean": float(np.mean(gap_values)),
            "event_gap_std": float(np.std(gap_values)),
            "event_gap_max": float(np.max(gap_values)),
            "event_gap_min": float(np.min(gap_values)),
            "unique_apps_opened": float(len(set(open_apps))),
            "switch_app_ratio": float(switch_count / max(seq_len, 1.0)),
            "click_link_ratio": float(click_count / max(seq_len, 1.0)),
            "scan_qr_ratio": float(qr_count / max(seq_len, 1.0)),
            "type_text_ratio": float(type_count / max(seq_len, 1.0)),
            "permission_action_ratio": float(perm_count / max(seq_len, 1.0)),
        }

    def _build_feature_table(self, merged: pd.DataFrame) -> pd.DataFrame:
        url_col = _find_column(list(merged.columns), ("url", "link_url", "payment_url", "upi_url"))
        text_col = _find_column(list(merged.columns), ("sms_text", "whatsapp_text", "message_text", "message", "text"))
        qr_col = _find_column(list(merged.columns), ("qr_data", "qr_payload", "qr_code"))

        excluded_encoded_cols = set(BLOCKED_ENCODED_COLUMNS)
        excluded_encoded_cols.update({"timestamp_start", "timestamp_end"})

        encoded_numeric_cols: list[str] = []
        for col in merged.columns:
            if col in excluded_encoded_cols:
                continue
            if col in BLOCKED_MODEL_FEATURE_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(merged[col]):
                encoded_numeric_cols.append(col)

        records: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            device_state = row.get("device_state_dict", {})
            behavioral = row.get("behavioral_dict", {})

            url_value = str(row[url_col]).strip() if url_col and pd.notna(row.get(url_col)) else ""
            if not url_value:
                url_value = self._synthesize_url(row)

            qr_value = str(row[qr_col]).strip() if qr_col and pd.notna(row.get(qr_col)) else ""
            if not qr_value:
                qr_value = self._synthesize_qr_data(row)

            text_value = str(row[text_col]).strip() if text_col and pd.notna(row.get(text_col)) else ""
            if not text_value:
                text_value = self._synthesize_message(row, url_value, qr_value)

            behavior_feats = self._extract_behavior_features(row)
            url_feats = self._extract_url_features(url_value)

            base_record: dict[str, Any] = {
                "session_id": str(row.get("session_id")),
                "label": str(row.get("label")).strip().lower(),
                "split": str(row.get("split")).strip(),
                "text_input": text_value,
                "url_input": url_value,
                "qr_data": qr_value,
                "scenario_family": str(row.get("scenario_family", "unknown")),
                "risk_score": _to_float(row.get("risk_score", 0.0)),
                "user_type": str(row.get("user_type", "unknown")),
                "vpn": _to_float(row.get("vpn", device_state.get("vpn", 0))),
                "rooted": _to_float(row.get("rooted", device_state.get("rooted", 0))),
                "emulator": _to_float(row.get("emulator", device_state.get("emulator", 0))),
                "ip_risk_score": _to_float(row.get("ip_risk_score", device_state.get("ip_risk_score", 0))),
                "session_duration": _to_float(row.get("session_duration", behavioral.get("session_duration", 0))),
                "num_events": _to_float(row.get("num_events", behavioral.get("num_events", 0))),
                "num_links_clicked": _to_float(row.get("num_links_clicked", behavioral.get("num_links_clicked", 0))),
                "num_qr_scans": _to_float(row.get("num_qr_scans", behavioral.get("num_qr_scans", 0))),
                "num_permission_requests": _to_float(
                    row.get("num_permission_requests", behavioral.get("num_permission_requests", 0))
                ),
            }
            base_record.update(behavior_feats)
            base_record.update(url_feats)

            for col in encoded_numeric_cols:
                if col in base_record:
                    continue
                base_record[col] = _to_float(row.get(col, 0.0))

            records.append(base_record)

        df = pd.DataFrame(records)
        df = df[df["label"].isin(LABEL_ORDER)].reset_index(drop=True)

        structured_exclude = {"label", "text_input", "url_input", "qr_data"}
        structured_exclude.update(BLOCKED_MODEL_FEATURE_COLUMNS)
        self.structured_cols = [col for col in df.columns if col not in structured_exclude]
        blocked_used = sorted(set(self.structured_cols).intersection(BLOCKED_MODEL_FEATURE_COLUMNS))
        if blocked_used:
            raise RuntimeError(f"Blocked feature columns leaked into structured features: {blocked_used}")
        # Robust dtype split: pandas string dtypes (e.g. `str`/`string`) should be treated as categorical.
        self.numeric_cols = [col for col in self.structured_cols if pd.api.types.is_numeric_dtype(df[col])]
        self.categorical_cols = [col for col in self.structured_cols if col not in self.numeric_cols]

        return df

    def _prepare_splits(self, feature_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        splits = {
            "train": feature_df[feature_df["split"] == "train"].reset_index(drop=True),
            "val": feature_df[feature_df["split"] == "val"].reset_index(drop=True),
            "test_ood": feature_df[feature_df["split"] == "test_ood"].reset_index(drop=True),
        }

        if self.config.quick_mode:
            quick_splits: dict[str, pd.DataFrame] = {}
            for split_name, split_df in splits.items():
                frac = 0.25 if split_name == "train" else 0.4
                chunks = []
                for _, group in split_df.groupby("label"):
                    n = max(10, int(len(group) * frac))
                    n = min(n, len(group))
                    chunks.append(group.sample(n=n, random_state=self.config.seed))
                sample = pd.concat(chunks, axis=0).sample(frac=1.0, random_state=self.config.seed).reset_index(drop=True)
                quick_splits[split_name] = sample
            splits = quick_splits

        return splits

    def _build_preprocessor(self) -> ColumnTransformer:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_cols),
                ("cat", categorical_pipeline, self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _label_encode(self, labels: pd.Series) -> np.ndarray:
        return np.array([self.label_to_id[str(label)] for label in labels], dtype=np.int64)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        report = classification_report(
            y_true,
            y_pred,
            labels=[self.label_to_id[label] for label in LABEL_ORDER],
            target_names=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
            "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
            "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
            "malicious_precision": float(report["malicious"]["precision"]),
            "malicious_recall": float(report["malicious"]["recall"]),
            "confusion_matrix": matrix,
        }

    def _load_transformer_backbone(self) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
        candidates = [self.config.transformer_model_name, self.config.fallback_transformer_model_name]
        last_err: Exception | None = None
        for name in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    name,
                    num_labels=len(LABEL_ORDER),
                    id2label={idx: label for idx, label in enumerate(LABEL_ORDER)},
                    label2id=self.label_to_id,
                )
                return tokenizer, model, name
            except Exception as err:
                last_err = err
                continue
        raise RuntimeError(f"Unable to load transformer model. Last error: {last_err}")

    def _trainer_compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        y_true = labels.astype(int)
        y_bin = (y_true == self.label_to_id["malicious"]).astype(int)
        p_bin = (preds == self.label_to_id["malicious"]).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "malicious_precision": float(precision_score(y_bin, p_bin, zero_division=0)),
            "malicious_recall": float(recall_score(y_bin, p_bin, zero_division=0)),
        }

    def _train_transformer(
        self,
        train_texts: list[str],
        val_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        tokenizer, _, selected_name = self._load_transformer_backbone()
        self.tokenizer = tokenizer
        self.transformer_model_name = selected_name

        train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=self.config.max_length)
        val_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=self.config.max_length)
        train_dataset = TextDataset(train_enc, y_train)
        val_dataset = TextDataset(val_enc, y_val)

        classes = np.array([self.label_to_id[label] for label in LABEL_ORDER], dtype=np.int64)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        best_state: dict[str, Any] | None = None
        best_metric = -1.0
        run_dirs: list[Path] = []

        for lr in self.config.transformer_learning_rates:
            model = AutoModelForSequenceClassification.from_pretrained(
                selected_name,
                num_labels=len(LABEL_ORDER),
                id2label={idx: label for idx, label in enumerate(LABEL_ORDER)},
                label2id=self.label_to_id,
            )
            run_dir = self.config.artifact_dir / f"tmp_transformer_lr_{str(lr).replace('.', '_')}"
            run_dirs.append(run_dir)
            arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
            kwargs: dict[str, Any] = {
                "output_dir": str(run_dir),
                "per_device_train_batch_size": self.config.train_batch_size,
                "per_device_eval_batch_size": self.config.eval_batch_size,
                "learning_rate": lr,
                "num_train_epochs": self.config.transformer_epochs,
                "weight_decay": self.config.transformer_weight_decay,
                "logging_strategy": "steps",
                "logging_steps": 100,
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "malicious_precision",
                "greater_is_better": True,
                "save_total_limit": 1,
                "report_to": [],
                "seed": self.config.seed,
                "dataloader_num_workers": 0,
            }
            if "evaluation_strategy" in arg_names:
                kwargs["evaluation_strategy"] = "epoch"
            if "eval_strategy" in arg_names:
                kwargs["eval_strategy"] = "epoch"
            training_args = TrainingArguments(**kwargs)
            trainer = WeightedLossTrainer(
                class_weights=weight_tensor,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._trainer_compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.transformer_patience)],
            )
            trainer.train()
            eval_metrics = trainer.evaluate()
            metric_value = float(eval_metrics.get("eval_malicious_precision", 0.0))
            if metric_value > best_metric:
                best_metric = metric_value
                best_state = model.state_dict()

        if best_state is None:
            raise RuntimeError("Transformer training did not produce a valid state.")

        final_model = AutoModelForSequenceClassification.from_pretrained(
            selected_name,
            num_labels=len(LABEL_ORDER),
            id2label={idx: label for idx, label in enumerate(LABEL_ORDER)},
            label2id=self.label_to_id,
        )
        final_model.load_state_dict(best_state)
        self.transformer_model = final_model

        for run_dir in run_dirs:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def _predict_transformer_proba(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if self.tokenizer is None or self.transformer_model is None:
            raise RuntimeError("Transformer model is not initialized.")
        if batch_size is None:
            batch_size = self.config.eval_batch_size

        self.transformer_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.transformer_model.to(device)

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def _extract_embeddings(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if self.tokenizer is None or self.transformer_model is None:
            raise RuntimeError("Transformer model is not initialized.")
        if batch_size is None:
            batch_size = self.config.eval_batch_size

        self.transformer_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.transformer_model.to(device)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embeddings.append(pooled.detach().cpu().numpy())

        return np.vstack(embeddings)

    def _train_xgb(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray,
    ) -> None:
        best_model: xgb.XGBClassifier | None = None
        best_precision = -1.0
        best_recall = -1.0
        mal_idx = self.label_to_id["malicious"]
        prefer_gpu = bool(torch.cuda.is_available())

        for params in self.config.xgb_param_grid:
            xgb_kwargs: dict[str, Any] = {
                "objective": "multi:softprob",
                "num_class": len(LABEL_ORDER),
                "n_estimators": self.config.xgb_estimators,
                "random_state": self.config.seed,
                "tree_method": "hist",
                "n_jobs": -1,
                "eval_metric": "mlogloss",
                "early_stopping_rounds": self.config.xgb_early_stopping_rounds,
            }
            if prefer_gpu:
                xgb_kwargs["device"] = "cuda"

            model = xgb.XGBClassifier(
                **xgb_kwargs,
                **params,
            )
            try:
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weights,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                )
            except xgb.core.XGBoostError:
                if not prefer_gpu:
                    raise
                prefer_gpu = False
                xgb_kwargs.pop("device", None)
                model = xgb.XGBClassifier(
                    **xgb_kwargs,
                    **params,
                )
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=sample_weights,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                )
            probs = model.predict_proba(x_val)
            preds = np.argmax(probs, axis=1)
            y_bin = (y_val == mal_idx).astype(int)
            p_bin = (preds == mal_idx).astype(int)
            precision = precision_score(y_bin, p_bin, zero_division=0)
            recall = recall_score(y_bin, p_bin, zero_division=0)
            if precision > best_precision or (math.isclose(precision, best_precision) and recall > best_recall):
                best_precision = precision
                best_recall = recall
                best_model = model

        if best_model is None:
            raise RuntimeError("XGBoost tuning did not produce a model.")
        self.xgb_model = best_model

    def _fit_isolation_forest(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        normal_idx = self.label_to_id["normal"]
        normal_mask = y_train == normal_idx
        if normal_mask.sum() < 20:
            normal_mask = np.ones_like(y_train, dtype=bool)

        iforest = IsolationForest(
            n_estimators=self.config.iso_estimators,
            contamination=self.config.iso_contamination,
            random_state=self.config.seed,
            n_jobs=-1,
        )
        iforest.fit(x_train[normal_mask])
        self.iforest = iforest

        train_raw = -iforest.decision_function(x_train)
        self.anomaly_scale_low = float(np.quantile(train_raw, 0.05))
        self.anomaly_scale_high = float(np.quantile(train_raw, 0.95))
        if math.isclose(self.anomaly_scale_low, self.anomaly_scale_high):
            self.anomaly_scale_high = self.anomaly_scale_low + 1.0

    def _anomaly_score(self, x_matrix: np.ndarray) -> np.ndarray:
        if self.iforest is None:
            raise RuntimeError("Isolation Forest not trained.")
        raw = -self.iforest.decision_function(x_matrix)
        score = (raw - self.anomaly_scale_low) / (self.anomaly_scale_high - self.anomaly_scale_low)
        return np.clip(score, 0.0, 1.0)

    def _ensemble(
        self,
        transformer_probs: np.ndarray,
        xgb_probs: np.ndarray,
        anomaly_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        w_t, w_x, w_a = self.config.ensemble_weights
        mal_idx = self.label_to_id["malicious"]
        normal_idx = self.label_to_id["normal"]

        malicious_score = w_t * transformer_probs[:, mal_idx] + w_x * xgb_probs[:, mal_idx] + w_a * anomaly_score
        combined = w_t * transformer_probs + w_x * xgb_probs
        combined[:, mal_idx] += w_a * anomaly_score
        combined[:, normal_idx] += w_a * (1.0 - anomaly_score)
        combined = np.clip(combined, 1e-9, None)
        combined = combined / combined.sum(axis=1, keepdims=True)

        return combined, malicious_score

    def _predict_labels_from_scores(self, combined_probs: np.ndarray, malicious_scores: np.ndarray) -> np.ndarray:
        normal_idx = self.label_to_id["normal"]
        suspicious_idx = self.label_to_id["suspicious"]
        malicious_idx = self.label_to_id["malicious"]

        pred = np.full(len(malicious_scores), normal_idx, dtype=np.int64)
        mal_mask = malicious_scores >= self.malicious_threshold
        pred[mal_mask] = malicious_idx

        not_mal = ~mal_mask
        suspicious_mask = np.logical_and(not_mal, combined_probs[:, suspicious_idx] >= self.suspicious_threshold)
        pred[suspicious_mask] = suspicious_idx

        fallback_mask = np.logical_and(not_mal, ~suspicious_mask)
        if fallback_mask.any():
            fallback_probs = combined_probs[fallback_mask][:, [normal_idx, suspicious_idx]]
            local_preds = np.argmax(fallback_probs, axis=1)
            pred[fallback_mask] = np.where(local_preds == 0, normal_idx, suspicious_idx)

        return pred

    def _tune_thresholds(self, y_val: np.ndarray, combined: np.ndarray, malicious_scores: np.ndarray) -> None:
        malicious_idx = self.label_to_id["malicious"]
        y_mal = (y_val == malicious_idx).astype(int)

        best_thr = self.malicious_threshold
        best_precision = -1.0
        best_f1 = -1.0
        for thr in np.linspace(0.45, 0.95, 51):
            preds = (malicious_scores >= thr).astype(int)
            precision = precision_score(y_mal, preds, zero_division=0)
            recall = recall_score(y_mal, preds, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if precision < 0.70:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_thr = float(thr)
        if best_f1 < 0:
            best_thr = float(np.quantile(malicious_scores, 0.85))
        self.malicious_threshold = best_thr

        best_susp_thr = self.suspicious_threshold
        best_macro = -1.0
        for thr in np.linspace(0.25, 0.75, 51):
            self.suspicious_threshold = float(thr)
            preds = self._predict_labels_from_scores(combined, malicious_scores)
            macro = classification_report(
                y_val,
                preds,
                labels=[0, 1, 2],
                target_names=LABEL_ORDER,
                output_dict=True,
                zero_division=0,
            )["macro avg"]["f1-score"]
            mal_prec = precision_score(y_mal, (preds == malicious_idx).astype(int), zero_division=0)
            if mal_prec >= best_precision * 0.97 and macro > best_macro:
                best_macro = macro
                best_susp_thr = float(thr)
        self.suspicious_threshold = best_susp_thr

    def _prepare_training_matrices(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        text_train = (train_df["text_input"] + " [URL] " + train_df["url_input"]).tolist()
        text_val = (val_df["text_input"] + " [URL] " + val_df["url_input"]).tolist()
        text_test = (test_df["text_input"] + " [URL] " + test_df["url_input"]).tolist()

        y_train = self._label_encode(train_df["label"])
        y_val = self._label_encode(val_df["label"])
        y_test = self._label_encode(test_df["label"])

        self._train_transformer(text_train, text_val, y_train, y_val)

        trans_train = self._predict_transformer_proba(text_train)
        trans_val = self._predict_transformer_proba(text_val)
        trans_test = self._predict_transformer_proba(text_test)

        emb_train = self._extract_embeddings(text_train)
        emb_val = self._extract_embeddings(text_val)
        emb_test = self._extract_embeddings(text_test)

        self.preprocessor = self._build_preprocessor()
        x_train_struct = self.preprocessor.fit_transform(train_df[self.structured_cols])
        x_val_struct = self.preprocessor.transform(val_df[self.structured_cols])
        x_test_struct = self.preprocessor.transform(test_df[self.structured_cols])

        self.structured_feature_names_out = list(self.preprocessor.get_feature_names_out())
        emb_names = [f"emb_{idx}" for idx in range(emb_train.shape[1])]
        self.full_feature_names = self.structured_feature_names_out + emb_names

        # Normalize embeddings using TRAINING stats only (avoid per-split leakage)
        self._emb_mean = emb_train.mean(axis=0)
        self._emb_std = emb_train.std(axis=0) + 1e-8
        emb_train_n = (emb_train - self._emb_mean) / self._emb_std
        emb_val_n = (emb_val - self._emb_mean) / self._emb_std
        emb_test_n = (emb_test - self._emb_mean) / self._emb_std

        x_train = np.hstack([x_train_struct, emb_train_n]).astype(np.float32)
        x_val = np.hstack([x_val_struct, emb_val_n]).astype(np.float32)
        x_test = np.hstack([x_test_struct, emb_test_n]).astype(np.float32)

        classes = np.array([self.label_to_id[label] for label in LABEL_ORDER], dtype=np.int64)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        sample_weights = np.array([class_weights[label] for label in y_train], dtype=np.float32)

        self.numeric_defaults = train_df[self.numeric_cols].median(numeric_only=True).to_dict()
        self.categorical_defaults = {}
        for col in self.categorical_cols:
            mode_vals = train_df[col].mode(dropna=True)
            self.categorical_defaults[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "unknown"

        return {
            "text_val": text_val,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "trans_val": trans_val,
            "trans_test": trans_test,
            "sample_weights": sample_weights,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        }

    def fit(self) -> dict[str, Any]:
        merged = self._load_data()
        feature_df = self._build_feature_table(merged)
        split_map = self._prepare_splits(feature_df)

        train_df = split_map["train"]
        val_df = split_map["val"]
        test_df = split_map["test_ood"]

        mats = self._prepare_training_matrices(train_df=train_df, val_df=val_df, test_df=test_df)
        self._train_xgb(
            mats["x_train"],
            mats["y_train"],
            mats["x_val"],
            mats["y_val"],
            mats["sample_weights"],
        )
        self._fit_isolation_forest(mats["x_train"], mats["y_train"])

        xgb_val = self.xgb_model.predict_proba(mats["x_val"])
        anom_val = self._anomaly_score(mats["x_val"])
        combined_val, mal_val = self._ensemble(mats["trans_val"], xgb_val, anom_val)
        self._tune_thresholds(mats["y_val"], combined_val, mal_val)

        val_pred = self._predict_labels_from_scores(combined_val, mal_val)
        val_metrics = self._compute_metrics(mats["y_val"], val_pred)

        xgb_test = self.xgb_model.predict_proba(mats["x_test"])
        anom_test = self._anomaly_score(mats["x_test"])
        combined_test, mal_test = self._ensemble(mats["trans_test"], xgb_test, anom_test)
        test_pred = self._predict_labels_from_scores(combined_test, mal_test)
        test_metrics = self._compute_metrics(mats["y_test"], test_pred)

        latency_info = self._export_and_benchmark_onnx(mats["text_val"][0] if mats["text_val"] else "upi verify", use_save=False)
        self.save_artifacts()

        example_predictions = self._build_example_predictions(mats["test_df"].head(12))
        example_path = self.config.artifact_dir / "example_predictions.csv"
        example_predictions.to_csv(example_path, index=False)

        report = _json_safe(
            {
            "config": asdict(self.config),
            "transformer_model_used": self.transformer_model_name,
            "class_order": LABEL_ORDER,
            "thresholds": {
                "malicious_threshold": self.malicious_threshold,
                "suspicious_threshold": self.suspicious_threshold,
            },
            "validation_metrics": val_metrics,
            "test_ood_metrics": test_metrics,
            "latency": latency_info,
            "artifact_paths": {
                "artifact_dir": str(self.config.artifact_dir.resolve()),
                "example_predictions": str(example_path.resolve()),
                "evaluation_report": str((self.config.artifact_dir / "evaluation_report.json").resolve()),
            },
            }
        )
        report_path = self.config.artifact_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _build_online_structured_row(
        self,
        input_text: str,
        url: str,
        qr_data: str,
        device_info: dict[str, Any] | None,
    ) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not loaded.")
        info = device_info or {}

        row: dict[str, Any] = {}
        for col in self.numeric_cols:
            row[col] = float(self.numeric_defaults.get(col, 0.0))
        for col in self.categorical_cols:
            row[col] = self.categorical_defaults.get(col, "unknown")

        row["scenario_family"] = str(info.get("scenario_family", row.get("scenario_family", "online_inference")))
        row["user_type"] = str(info.get("user_type", row.get("user_type", "unknown_user")))
        row["risk_score"] = _to_float(info.get("risk_score", row.get("risk_score", 0.5)))
        row["vpn"] = _to_float(info.get("vpn", row.get("vpn", 0.0)))
        row["rooted"] = _to_float(info.get("rooted", row.get("rooted", 0.0)))
        row["emulator"] = _to_float(info.get("emulator", row.get("emulator", 0.0)))
        row["ip_risk_score"] = _to_float(info.get("ip_risk_score", row.get("ip_risk_score", 0.4)))

        inferred_events = max(len(input_text.split()), 1)
        row["num_events"] = _to_float(info.get("num_events", inferred_events))
        row["num_links_clicked"] = _to_float(info.get("num_links_clicked", 1 if url else 0))
        row["num_qr_scans"] = _to_float(info.get("num_qr_scans", 1 if qr_data else 0))
        row["num_permission_requests"] = _to_float(info.get("num_permission_requests", row.get("num_permission_requests", 0)))
        row["session_duration"] = _to_float(info.get("session_duration", row.get("session_duration", max(inferred_events * 6, 30))))
        row["sequence_length"] = _to_float(info.get("sequence_length", row.get("num_events", inferred_events)))
        row["event_gap_mean"] = _to_float(info.get("event_gap_mean", row.get("session_duration", 60.0) / max(row["sequence_length"], 1.0)))
        row["event_gap_std"] = _to_float(info.get("event_gap_std", row.get("event_gap_mean", 8.0) * 0.35))
        row["event_gap_max"] = _to_float(info.get("event_gap_max", row.get("event_gap_mean", 8.0) * 2.0))
        row["event_gap_min"] = _to_float(info.get("event_gap_min", max(row.get("event_gap_mean", 8.0) * 0.2, 0.5)))

        # Pass through ALL device_info keys that match known feature columns
        for key, value in info.items():
            if key in self.numeric_cols and key not in ("vpn", "rooted", "emulator", "ip_risk_score",
                "num_events", "num_links_clicked", "num_qr_scans", "num_permission_requests",
                "session_duration", "sequence_length", "event_gap_mean", "event_gap_std",
                "event_gap_max", "event_gap_min"):
                row[key] = _to_float(value)

        url_feats = self._extract_url_features(url or "https://unknown.local")
        for key, value in url_feats.items():
            if key in row:
                row[key] = value

        safe_row = {col: row.get(col, self.numeric_defaults.get(col, self.categorical_defaults.get(col, 0))) for col in self.structured_cols}
        return pd.DataFrame([safe_row], columns=self.structured_cols)

    def _online_text(self, input_text: str, url: str, qr_data: str, device_info: dict[str, Any] | None) -> str:
        info = device_info or {}
        device_part = f"vpn={int(_to_float(info.get('vpn', 0)))} rooted={int(_to_float(info.get('rooted', 0)))} emulator={int(_to_float(info.get('emulator', 0)))}"
        scenario = str(info.get("scenario_family", "online_inference"))
        qr_part = f" QR payload: {qr_data}." if qr_data else ""
        return f"{input_text} URL: {url}. Scenario {scenario}. Device {device_part}.{qr_part}"

    def _xgb_contributions(self, x_matrix: np.ndarray) -> np.ndarray:
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not loaded.")
        dmatrix = xgb.DMatrix(x_matrix, feature_names=self.full_feature_names)
        contrib = self.xgb_model.get_booster().predict(dmatrix, pred_contribs=True)
        arr = np.array(contrib)
        malicious_idx = self.label_to_id["malicious"]

        if arr.ndim == 3:
            return arr[0, malicious_idx, :-1]
        if arr.ndim == 2:
            block = arr.shape[1] // len(LABEL_ORDER)
            start = malicious_idx * block
            end = start + block - 1
            return arr[0, start:end]
        raise RuntimeError("Unexpected XGBoost contribution shape.")

    def _format_explanations(self, x_matrix: np.ndarray, online_row: pd.DataFrame) -> list[str]:
        try:
            contrib = self._xgb_contributions(x_matrix)
        except Exception:
            contrib = np.zeros(len(self.full_feature_names), dtype=np.float32)

        names = np.array(self.full_feature_names)
        mask_structured = np.array([not name.startswith("emb_") for name in names])
        contrib_struct = contrib[mask_structured]
        name_struct = names[mask_structured]

        ranked_idx = np.argsort(np.abs(contrib_struct))[::-1]
        explanations: list[str] = []
        for idx in ranked_idx[:4]:
            raw_name = name_struct[idx]
            raw_name = raw_name.replace("num__", "").replace("cat__", "").replace("onehot__", "")
            explanations.append(f"{raw_name}: contribution {contrib_struct[idx]:+.4f}")

        row = online_row.iloc[0].to_dict()
        if _to_float(row.get("rooted", 0)) > 0:
            explanations.append("rooted device flag detected")
        if _to_float(row.get("vpn", 0)) > 0:
            explanations.append("vpn usage observed")
        if _to_float(row.get("url_keyword_count", 0)) >= 2:
            explanations.append("multiple risky URL keywords found")

        unique: list[str] = []
        for item in explanations:
            if item not in unique:
                unique.append(item)
        return unique[:5]

    def detect_fraud(
        self,
        input_text: str,
        url: str,
        qr_data: str,
        device_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.preprocessor is None or self.tokenizer is None or self.transformer_model is None or self.xgb_model is None:
            raise RuntimeError("Model artifacts are not loaded.")

        online_text = self._online_text(input_text=input_text, url=url, qr_data=qr_data, device_info=device_info)
        transformer_probs = self._predict_transformer_proba([online_text])
        embedding = self._extract_embeddings([online_text])

        online_structured_df = self._build_online_structured_row(
            input_text=input_text,
            url=url,
            qr_data=qr_data,
            device_info=device_info,
        )
        x_structured = self.preprocessor.transform(online_structured_df[self.structured_cols])
        # Normalize embeddings using stored training stats for consistency
        if self._emb_mean is not None and self._emb_std is not None:
            embedding = (embedding - self._emb_mean) / self._emb_std
        x_all = np.hstack([x_structured, embedding]).astype(np.float32)

        xgb_probs = self.xgb_model.predict_proba(x_all)
        anomaly = self._anomaly_score(x_all)
        combined, malicious_scores = self._ensemble(transformer_probs, xgb_probs, anomaly)
        pred = self._predict_labels_from_scores(combined, malicious_scores)[0]

        predicted_label = self.id_to_label[int(pred)]
        fraud_prob = float(np.clip(malicious_scores[0], 0.0, 1.0))
        # Risk level tied to predicted label for intuitive results
        risk_level = {"malicious": "HIGH", "suspicious": "MEDIUM", "normal": "LOW"}[predicted_label]
        explanation = self._format_explanations(x_all, online_structured_df)

        return {
            "predicted_label": predicted_label,
            "risk_level": risk_level,
            "fraud_probability": fraud_prob,
            "class_probabilities": {label: float(combined[0, idx]) for idx, label in enumerate(LABEL_ORDER)},
            "explanation": explanation,
        }

    def predict_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("Model artifacts are not loaded.")

        texts = (df["text_input"] + " [URL] " + df["url_input"]).tolist()
        transformer_probs = self._predict_transformer_proba(texts)
        embeddings = self._extract_embeddings(texts)

        x_struct = self.preprocessor.transform(df[self.structured_cols])
        if self._emb_mean is not None and self._emb_std is not None:
            embeddings = (embeddings - self._emb_mean) / self._emb_std
        x_all = np.hstack([x_struct, embeddings]).astype(np.float32)
        xgb_probs = self.xgb_model.predict_proba(x_all)
        anomaly = self._anomaly_score(x_all)
        combined, malicious_scores = self._ensemble(transformer_probs, xgb_probs, anomaly)
        pred = self._predict_labels_from_scores(combined, malicious_scores)

        out = df[["session_id", "label", "split", "scenario_family", "risk_score"]].copy()
        out["predicted_label"] = [self.id_to_label[int(p)] for p in pred]
        out["fraud_probability"] = malicious_scores
        label_risk_map = {"malicious": "HIGH", "suspicious": "MEDIUM", "normal": "LOW"}
        out["risk_level"] = out["predicted_label"].map(label_risk_map)
        return out

    def _ensure_onnx_available(self) -> bool:
        try:
            import onnx  # noqa: F401

            return True
        except Exception:
            pass

        candidates = []
        env_candidate = os.environ.get("ONNX_VENDOR_PATH", "").strip()
        if env_candidate:
            candidates.append(env_candidate)
        candidates.extend(self.config.onnx_vendor_paths)

        for path in candidates:
            if not path:
                continue
            candidate = Path(path)
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                try:
                    import onnx  # noqa: F401

                    return True
                except Exception:
                    continue
        return False

    def _export_transformer_to_onnx(self, onnx_path: Path) -> bool:
        if self.tokenizer is None or self.transformer_model is None:
            return False
        if not self._ensure_onnx_available():
            return False

        original_device = next(self.transformer_model.parameters()).device
        export_model = self.transformer_model.to("cpu").eval()
        wrapper = HFClassifierWrapper(export_model)
        dummy = self.tokenizer(
            "UPI account verification required",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.onnx.export(
                wrapper,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=self.config.onnx_opset,
                do_constant_folding=True,
                dynamo=False,
            )
            return True
        except Exception:
            return False
        finally:
            self.transformer_model.to(original_device)

    def _benchmark_onnx(self, onnx_path: Path, sample_text: str) -> dict[str, Any]:
        try:
            import onnxruntime as ort
        except Exception:
            return {"available": False, "reason": "onnxruntime_not_installed"}

        if self.tokenizer is None:
            return {"available": False, "reason": "tokenizer_missing"}
        if not onnx_path.exists():
            return {"available": False, "reason": "onnx_file_missing"}

        enc = self.tokenizer(
            sample_text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

        for _ in range(10):
            _ = session.run(None, inputs)

        started = time.perf_counter()
        for _ in range(self.config.latency_runs):
            _ = session.run(None, inputs)
        elapsed = time.perf_counter() - started
        avg_ms = (elapsed / self.config.latency_runs) * 1000.0

        return {
            "available": True,
            "onnx_path": str(onnx_path.resolve()),
            "avg_latency_ms": avg_ms,
            "target_met_lt_200ms": bool(avg_ms < 200.0),
        }

    def _export_and_benchmark_onnx(self, sample_text: str, use_save: bool = True) -> dict[str, Any]:
        onnx_path = self.config.artifact_dir / "onnx" / "transformer_classifier.onnx"
        onnx_ok = self._export_transformer_to_onnx(onnx_path)
        if not onnx_ok:
            return {
                "onnx_exported": False,
                "reason": "onnx_dependency_missing",
                "target_met_lt_200ms": False,
            }

        latency_info = self._benchmark_onnx(onnx_path, sample_text)
        out = {"onnx_exported": True, **latency_info}

        if use_save:
            latency_path = self.config.artifact_dir / "onnx_latency.json"
            latency_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    def _build_example_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            pred = self.detect_fraud(
                input_text=str(row["text_input"]),
                url=str(row["url_input"]),
                qr_data=str(row["qr_data"]),
                device_info={
                    "vpn": int(_to_float(row.get("vpn", 0))),
                    "rooted": int(_to_float(row.get("rooted", 0))),
                    "emulator": int(_to_float(row.get("emulator", 0))),
                    "scenario_family": str(row.get("scenario_family", "unknown")),
                    "risk_score": _to_float(row.get("risk_score", 0.5)),
                    "num_events": _to_float(row.get("num_events", 0)),
                    "num_links_clicked": _to_float(row.get("num_links_clicked", 0)),
                    "num_qr_scans": _to_float(row.get("num_qr_scans", 0)),
                },
            )
            rows.append(
                {
                    "session_id": row["session_id"],
                    "true_label": row["label"],
                    "predicted_label": pred["predicted_label"],
                    "risk_level": pred["risk_level"],
                    "fraud_probability": pred["fraud_probability"],
                    "top_explanation": " | ".join(pred["explanation"]),
                }
            )
        return pd.DataFrame(rows)

    def save_artifacts(self) -> None:
        if self.tokenizer is None or self.transformer_model is None or self.preprocessor is None or self.xgb_model is None:
            raise RuntimeError("Cannot save incomplete model artifacts.")

        self.config.ensure_dirs()
        transformer_dir = self.config.artifact_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(transformer_dir))
        self.transformer_model.save_pretrained(str(transformer_dir))

        joblib.dump(self.preprocessor, self.config.artifact_dir / "structured_preprocessor.joblib")
        joblib.dump(self.iforest, self.config.artifact_dir / "isolation_forest.joblib")
        self.xgb_model.save_model(str(self.config.artifact_dir / "xgboost_model.json"))

        metadata = _json_safe(
            {
            "class_order": LABEL_ORDER,
            "label_to_id": self.label_to_id,
            "transformer_model_name": self.transformer_model_name,
            "max_length": self.config.max_length,
            "structured_cols": self.structured_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "structured_feature_names_out": self.structured_feature_names_out,
            "full_feature_names": self.full_feature_names,
            "numeric_defaults": self.numeric_defaults,
            "categorical_defaults": self.categorical_defaults,
            "anomaly_scale_low": self.anomaly_scale_low,
            "anomaly_scale_high": self.anomaly_scale_high,
            "malicious_threshold": self.malicious_threshold,
            "suspicious_threshold": self.suspicious_threshold,
            "ensemble_weights": self.config.ensemble_weights,
            "emb_mean": self._emb_mean.tolist() if self._emb_mean is not None else None,
            "emb_std": self._emb_std.tolist() if self._emb_std is not None else None,
            "seed": self.config.seed,
            }
        )
        (self.config.artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self._export_and_benchmark_onnx("UPI verification message with payment link", use_save=True)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "HybridFraudDetector":
        artifact_path = Path(artifact_dir)
        metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
        config = PipelineConfig(artifact_dir=artifact_path, seed=int(metadata.get("seed", 42)))
        if "max_length" in metadata:
            try:
                config.max_length = int(metadata["max_length"])
            except (TypeError, ValueError):
                pass
        raw_weights = metadata.get("ensemble_weights")
        if isinstance(raw_weights, (list, tuple)) and len(raw_weights) == 3:
            try:
                config.ensemble_weights = tuple(float(v) for v in raw_weights)
            except (TypeError, ValueError):
                pass
        model = cls(config=config)

        model.label_to_id = {k: int(v) for k, v in metadata["label_to_id"].items()}
        model.id_to_label = {int(v): k for k, v in metadata["label_to_id"].items()}
        model.transformer_model_name = metadata["transformer_model_name"]

        transformer_dir = artifact_path / "transformer"
        model.tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))
        model.transformer_model = AutoModelForSequenceClassification.from_pretrained(str(transformer_dir))
        model.preprocessor = joblib.load(artifact_path / "structured_preprocessor.joblib")
        model.iforest = joblib.load(artifact_path / "isolation_forest.joblib")
        model.xgb_model = xgb.XGBClassifier()
        model.xgb_model.load_model(str(artifact_path / "xgboost_model.json"))

        model.structured_cols = list(metadata["structured_cols"])
        model.numeric_cols = list(metadata["numeric_cols"])
        model.categorical_cols = list(metadata["categorical_cols"])
        model.structured_feature_names_out = list(metadata["structured_feature_names_out"])
        model.full_feature_names = list(metadata["full_feature_names"])
        model.numeric_defaults = {k: float(v) for k, v in metadata["numeric_defaults"].items()}
        model.categorical_defaults = {k: str(v) for k, v in metadata["categorical_defaults"].items()}
        model.anomaly_scale_low = float(metadata["anomaly_scale_low"])
        model.anomaly_scale_high = float(metadata["anomaly_scale_high"])
        model.malicious_threshold = float(metadata["malicious_threshold"])
        model.suspicious_threshold = float(metadata["suspicious_threshold"])

        # Load embedding normalization stats
        emb_mean_raw = metadata.get("emb_mean")
        emb_std_raw = metadata.get("emb_std")
        if emb_mean_raw is not None and emb_std_raw is not None:
            model._emb_mean = np.array(emb_mean_raw, dtype=np.float32)
            model._emb_std = np.array(emb_std_raw, dtype=np.float32)

        return model


def train_hybrid_detector(config: PipelineConfig | None = None) -> dict[str, Any]:
    detector = HybridFraudDetector(config=config)
    report = detector.fit()
    return report
