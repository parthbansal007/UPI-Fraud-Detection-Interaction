from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fraud_system.hybrid_pipeline import HybridFraudDetector, PipelineConfig


@dataclass
class FeatureMatrixConfig:
    data_dir: Path = Path("interaction_data")
    transformer_dir: Path = Path("models/interaction/transformer")
    output_dir: Path = Path("outputs/processed_matrices")
    seed: int = 42
    max_length: int | None = None
    batch_size: int = 32


def _resolve_max_length(config: FeatureMatrixConfig) -> int:
    if config.max_length is not None:
        return int(config.max_length)

    report_path = config.transformer_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        value = report.get("config", {}).get("max_length")
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, float) and value > 0:
            return int(value)

    return 96


def _prepare_feature_splits(config: FeatureMatrixConfig) -> tuple[HybridFraudDetector, dict[str, pd.DataFrame]]:
    pipeline_cfg = PipelineConfig(
        seed=config.seed,
        raw_data_path=config.data_dir / "dataset_v2.csv",
        encoded_data_path=config.data_dir / "dataset_encoded_v2.csv",
        artifact_dir=config.transformer_dir.parent,
    )
    detector = HybridFraudDetector(config=pipeline_cfg)
    merged = detector._load_data()
    feature_df = detector._build_feature_table(merged)
    split_map = detector._prepare_splits(feature_df)

    for split_name in ("train", "val", "test_ood"):
        if split_map[split_name].empty:
            raise ValueError(f"Split '{split_name}' is empty; cannot build feature matrices.")

    return detector, split_map


def _to_text(df: pd.DataFrame) -> list[str]:
    return (df["text_input"].astype(str) + " [URL] " + df["url_input"].astype(str)).tolist()


def _encode_labels(df: pd.DataFrame, label_to_id: dict[str, int]) -> np.ndarray:
    labels = df["label"].astype(str).str.strip().str.lower().tolist()
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def _to_float_matrix(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _extract_embeddings(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.detach().cpu().numpy())

    if not embeddings:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32)


def _assert_no_nan(name: str, arr: np.ndarray) -> None:
    if np.isnan(arr).any():
        raise ValueError(f"NaN values found in matrix '{name}'.")


def generate_transformer_feature_matrices(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = FeatureMatrixConfig(**(config or {}))
    cfg.data_dir = Path(cfg.data_dir)
    cfg.transformer_dir = Path(cfg.transformer_dir)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.transformer_dir.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {cfg.transformer_dir}")

    max_length = _resolve_max_length(cfg)

    detector, split_map = _prepare_feature_splits(cfg)
    train_df = split_map["train"].reset_index(drop=True)
    val_df = split_map["val"].reset_index(drop=True)
    test_df = split_map["test_ood"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.transformer_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(cfg.transformer_dir))

    structured_cols = detector.structured_cols
    preprocessor = detector._build_preprocessor()
    x_train_struct = _to_float_matrix(preprocessor.fit_transform(train_df[structured_cols]))
    x_val_struct = _to_float_matrix(preprocessor.transform(val_df[structured_cols]))
    x_test_struct = _to_float_matrix(preprocessor.transform(test_df[structured_cols]))

    train_texts = _to_text(train_df)
    val_texts = _to_text(val_df)
    test_texts = _to_text(test_df)
    emb_train = _extract_embeddings(model, tokenizer, train_texts, max_length=max_length, batch_size=cfg.batch_size)
    emb_val = _extract_embeddings(model, tokenizer, val_texts, max_length=max_length, batch_size=cfg.batch_size)
    emb_test = _extract_embeddings(model, tokenizer, test_texts, max_length=max_length, batch_size=cfg.batch_size)

    X_train = np.hstack([x_train_struct, emb_train]).astype(np.float32)
    X_val = np.hstack([x_val_struct, emb_val]).astype(np.float32)
    X_test = np.hstack([x_test_struct, emb_test]).astype(np.float32)

    y_train = _encode_labels(train_df, detector.label_to_id)
    y_val = _encode_labels(val_df, detector.label_to_id)
    y_test = _encode_labels(test_df, detector.label_to_id)

    _assert_no_nan("X_train", X_train)
    _assert_no_nan("X_val", X_val)
    _assert_no_nan("X_test", X_test)

    matrix_paths = {
        "X_train": cfg.output_dir / "X_train.npy",
        "X_val": cfg.output_dir / "X_val.npy",
        "X_test": cfg.output_dir / "X_test.npy",
        "y_train": cfg.output_dir / "y_train.npy",
        "y_val": cfg.output_dir / "y_val.npy",
        "y_test": cfg.output_dir / "y_test.npy",
        "bundle": cfg.output_dir / "feature_matrices.npz",
        "row_index": cfg.output_dir / "row_index.csv",
        "feature_names": cfg.output_dir / "feature_names.json",
        "report": cfg.output_dir / "feature_matrix_report.json",
    }

    np.save(matrix_paths["X_train"], X_train)
    np.save(matrix_paths["X_val"], X_val)
    np.save(matrix_paths["X_test"], X_test)
    np.save(matrix_paths["y_train"], y_train)
    np.save(matrix_paths["y_val"], y_val)
    np.save(matrix_paths["y_test"], y_test)
    np.savez_compressed(
        matrix_paths["bundle"],
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    row_index = pd.concat(
        [
            pd.DataFrame({"split": "train", "session_id": train_df["session_id"], "label": train_df["label"]}),
            pd.DataFrame({"split": "val", "session_id": val_df["session_id"], "label": val_df["label"]}),
            pd.DataFrame({"split": "test_ood", "session_id": test_df["session_id"], "label": test_df["label"]}),
        ],
        axis=0,
        ignore_index=True,
    )
    row_index.to_csv(matrix_paths["row_index"], index=False)

    structured_feature_names = list(preprocessor.get_feature_names_out())
    embedding_feature_names = [f"emb_{idx}" for idx in range(int(emb_train.shape[1]))]
    feature_name_payload = {
        "structured_feature_names": structured_feature_names,
        "embedding_feature_names": embedding_feature_names,
        "full_feature_names": structured_feature_names + embedding_feature_names,
    }
    matrix_paths["feature_names"].write_text(json.dumps(feature_name_payload, indent=2), encoding="utf-8")

    report = {
        "device_used": "cuda" if torch.cuda.is_available() else "cpu",
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test_ood": int(len(test_df))},
        "matrix_shapes": {
            "x_train_structured": list(x_train_struct.shape),
            "x_val_structured": list(x_val_struct.shape),
            "x_test_structured": list(x_test_struct.shape),
            "emb_train": list(emb_train.shape),
            "emb_val": list(emb_val.shape),
            "emb_test": list(emb_test.shape),
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "max_length": int(max_length),
        "batch_size": int(cfg.batch_size),
        "label_to_id": detector.label_to_id,
        "artifact_paths": {key: str(path.resolve()) for key, path in matrix_paths.items()},
        "config": {
            **asdict(cfg),
            "data_dir": str(cfg.data_dir),
            "transformer_dir": str(cfg.transformer_dir),
            "output_dir": str(cfg.output_dir),
        },
    }
    matrix_paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
