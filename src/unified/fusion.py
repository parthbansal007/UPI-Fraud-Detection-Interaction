from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.interaction import predict_interaction_risk
from src.transaction import predict_transaction_risk

DEFAULT_FUSION_CONFIG_PATH = Path("models/unified/fusion_config.json")


@dataclass
class FusionConfig:
    interaction_weight: float = 0.55
    transaction_weight: float = 0.45
    high_threshold: float = 0.70
    medium_threshold: float = 0.40
    interaction_model_dir: str = "models/interaction"
    transaction_model_dir: str = "models/transaction"


def _ensure_dataframe(df_or_inputs: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    if isinstance(df_or_inputs, pd.DataFrame):
        return df_or_inputs.copy()
    if isinstance(df_or_inputs, dict):
        return pd.DataFrame([df_or_inputs])
    if isinstance(df_or_inputs, list):
        return pd.DataFrame([dict(row) for row in df_or_inputs])
    raise TypeError("df_or_inputs must be a DataFrame, dict, or list[dict].")


def load_fusion_config(path: str | Path = DEFAULT_FUSION_CONFIG_PATH) -> FusionConfig:
    config_path = Path(path)
    if not config_path.exists():
        config = FusionConfig()
        save_fusion_config(config, path=config_path)
        return config

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return FusionConfig(**payload)


def save_fusion_config(config: FusionConfig, path: str | Path = DEFAULT_FUSION_CONFIG_PATH) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return config_path


def _fused_level(score: float, config: FusionConfig) -> str:
    if score >= config.high_threshold:
        return "HIGH"
    if score >= config.medium_threshold:
        return "MEDIUM"
    return "LOW"


def train_unified_fusion(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = FusionConfig(**(config or {}))
    config_path = save_fusion_config(cfg)
    return {
        "status": "configured",
        "config_path": str(config_path.resolve()),
        "config": asdict(cfg),
    }


def infer_unified_risk(
    df_or_inputs: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    base_df = _ensure_dataframe(df_or_inputs)
    cfg = FusionConfig(**config) if config else load_fusion_config()

    interaction_df = predict_interaction_risk(base_df, model_dir=cfg.interaction_model_dir)
    transaction_df = predict_transaction_risk(base_df, model_dir=cfg.transaction_model_dir)

    if len(interaction_df) != len(transaction_df):
        raise RuntimeError("Interaction and transaction predictions returned mismatched row counts.")

    interaction_score = pd.to_numeric(interaction_df["interaction_risk_score"], errors="coerce").fillna(0.0)
    transaction_score = pd.to_numeric(transaction_df["transaction_score"], errors="coerce").fillna(0.0)

    fused_score = (
        cfg.interaction_weight * interaction_score.to_numpy(dtype=float)
        + cfg.transaction_weight * transaction_score.to_numpy(dtype=float)
    )
    fused_level = [_fused_level(float(score), cfg) for score in fused_score]

    session_id = interaction_df.get("session_id")
    if session_id is None:
        session_id = transaction_df.get("session_id")
    if session_id is None:
        session_id = pd.Series([f"session_{idx:06d}" for idx in range(len(base_df))])

    out = pd.DataFrame(
        {
            "session_id": session_id.astype(str),
            "interaction_score": interaction_score.astype(float),
            "transaction_score": transaction_score.astype(float),
            "fused_risk_score": fused_score,
            "fused_risk_level": fused_level,
            "interaction_label": interaction_df.get("interaction_label", pd.Series(["unknown"] * len(base_df))).astype(str),
            "transaction_label": transaction_df.get("transaction_label", pd.Series(["unknown"] * len(base_df))).astype(str),
        }
    )
    out["per_model_explanations"] = out.apply(
        lambda row: json.dumps(
            {
                "interaction": {
                    "score": float(row["interaction_score"]),
                    "weight": cfg.interaction_weight,
                    "label": row["interaction_label"],
                },
                "transaction": {
                    "score": float(row["transaction_score"]),
                    "weight": cfg.transaction_weight,
                    "label": row["transaction_label"],
                },
            }
        ),
        axis=1,
    )

    if "true_label" in interaction_df.columns:
        out["true_label"] = interaction_df["true_label"].astype(str)
    elif "true_label" in transaction_df.columns:
        out["true_label"] = transaction_df["true_label"].astype(str)
    elif "fraud_flag" in base_df.columns:
        out["true_label"] = pd.to_numeric(base_df["fraud_flag"], errors="coerce").fillna(0).astype(int)

    return out
