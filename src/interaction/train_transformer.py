from __future__ import annotations

import inspect
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, TrainingArguments

from fraud_system.hybrid_pipeline import (
    LABEL_ORDER,
    HybridFraudDetector,
    PipelineConfig,
    TextDataset,
    WeightedLossTrainer,
)


@dataclass
class TransformerTextConfig:
    data_dir: Path = Path("interaction_data")
    model_dir: Path = Path("models/interaction/transformer")
    output_dir: Path = Path("outputs")
    seed: int = 42
    model_candidates: tuple[str, ...] = ("distilbert-base-uncased",)
    max_length: int = 96
    train_batch_size: int = 16
    eval_batch_size: int = 32
    tokenization_batch_size: int = 256
    epochs: float = 4.0
    learning_rates: tuple[float, ...] = (2e-5, 3e-5)
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    class_weights: tuple[float, float, float] = (1.0, 2.0, 4.0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_backbone(
    model_candidates: tuple[str, ...],
    label_to_id: dict[str, int],
) -> tuple[AutoTokenizer, str]:
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    last_err: Exception | None = None
    for model_name in model_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            _ = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(LABEL_ORDER),
                id2label=id_to_label,
                label2id=label_to_id,
            )
            return tokenizer, model_name
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Unable to load transformer backbone. Last error: {last_err}")


def _tokenize_batched(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> dict[str, list[list[int]]]:
    encoded: dict[str, list[list[int]]] = {}
    for start in range(0, len(texts), batch_size):
        print(f"Tokenizing batch from index: {start}")
        chunk = texts[start : start + batch_size]
        chunk_enc = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        for key, values in chunk_enc.items():
            encoded.setdefault(key, []).extend(values)
    return encoded


def _prepare_text_splits(config: TransformerTextConfig) -> dict[str, pd.DataFrame]:
    pipeline_cfg = PipelineConfig(
        seed=config.seed,
        raw_data_path=config.data_dir / "dataset_v2.csv",
        encoded_data_path=config.data_dir / "dataset_encoded_v2.csv",
        artifact_dir=config.model_dir.parent,
    )
    detector = HybridFraudDetector(config=pipeline_cfg)
    merged = detector._load_data()
    feature_df = detector._build_feature_table(merged)
    split_map = detector._prepare_splits(feature_df)

    for split_name in ("train", "val"):
        if split_map[split_name].empty:
            raise ValueError(f"Split '{split_name}' is empty; cannot train transformer model.")
    return split_map


def _to_text(df: pd.DataFrame) -> list[str]:
    df["text_input"] = df["text_input"].fillna("").astype(str)
    df["url_input"] = df["url_input"].fillna("https://unknown.local").astype(str)

    texts = (df["text_input"] + " [URL] " + df["url_input"]).tolist()

    # LIMIT TEXT LENGTH (VERY IMPORTANT)
    texts = [t[:300] for t in texts]

    return texts


def _encode_labels(df: pd.DataFrame, label_to_id: dict[str, int]) -> np.ndarray:
    labels = df["label"].astype(str).str.strip().str.lower().tolist()
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    y_bin = (y_true == 2).astype(int)
    p_bin = (y_pred == 2).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "malicious_precision": float(precision_score(y_bin, p_bin, zero_division=0)),
        "malicious_recall": float(recall_score(y_bin, p_bin, zero_division=0)),
        "precision_by_class": {label: float(report[label]["precision"]) for label in LABEL_ORDER},
        "recall_by_class": {label: float(report[label]["recall"]) for label in LABEL_ORDER},
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in LABEL_ORDER},
    }


def _trainer_compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    y_true = labels.astype(int)
    y_bin = (y_true == 2).astype(int)
    p_bin = (preds == 2).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "malicious_precision": float(precision_score(y_bin, p_bin, zero_division=0)),
        "malicious_recall": float(recall_score(y_bin, p_bin, zero_division=0)),
    }


def _predict_probabilities(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs: list[np.ndarray] = []
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
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def _build_scores_table(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    pred_ids = np.argmax(probs, axis=1)
    pred_labels = [LABEL_ORDER[idx] for idx in pred_ids]
    return pd.DataFrame(
        {
            "session_id": df["session_id"].astype(str).values,
            "split": df["split"].astype(str).values,
            "true_label": df["label"].astype(str).values,
            "predicted_label": pred_labels,
            "normal_probability": probs[:, 0],
            "suspicious_probability": probs[:, 1],
            "malicious_probability": probs[:, 2],
        }
    )


def train_transformer_text_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = TransformerTextConfig(**(config or {}))
    cfg.data_dir = Path(cfg.data_dir)
    cfg.model_dir = Path(cfg.model_dir)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)

    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    split_map = _prepare_text_splits(cfg)
    train_df = split_map["train"].reset_index(drop=True)
    val_df = split_map["val"].reset_index(drop=True)
    test_df = split_map["test_ood"].reset_index(drop=True)

    train_texts = _to_text(train_df)
    val_texts = _to_text(val_df)
    test_texts = _to_text(test_df)
    y_train = _encode_labels(train_df, label_to_id)
    y_val = _encode_labels(val_df, label_to_id)
    y_test = _encode_labels(test_df, label_to_id)

    tokenizer, selected_model_name = _load_backbone(cfg.model_candidates, label_to_id)
    train_enc = _tokenize_batched(tokenizer, train_texts, cfg.max_length, cfg.tokenization_batch_size)
    val_enc = _tokenize_batched(tokenizer, val_texts, cfg.max_length, cfg.tokenization_batch_size)
    train_dataset = TextDataset(train_enc, y_train)
    val_dataset = TextDataset(val_enc, y_val)

    weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32)
    run_dirs: list[Path] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -1.0
    best_lr = cfg.learning_rates[0]

    for lr in cfg.learning_rates:
        model = AutoModelForSequenceClassification.from_pretrained(
            selected_model_name,
            num_labels=len(LABEL_ORDER),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        run_dir = cfg.model_dir.parent / f"tmp_text_transformer_lr_{str(lr).replace('.', '_')}"
        run_dirs.append(run_dir)
        arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        kwargs: dict[str, Any] = {
            "output_dir": str(run_dir),
            "per_device_train_batch_size": cfg.train_batch_size,
            "per_device_eval_batch_size": cfg.eval_batch_size,
            "learning_rate": lr,
            "num_train_epochs": cfg.epochs,
            "weight_decay": cfg.weight_decay,
            "logging_strategy": "steps",
            "logging_steps": 100,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "malicious_precision",
            "greater_is_better": True,
            "save_total_limit": 1,
            "report_to": [],
            "seed": cfg.seed,
            "dataloader_num_workers": 0,
        }
        if "evaluation_strategy" in arg_names:
            kwargs["evaluation_strategy"] = "epoch"
        if "eval_strategy" in arg_names:
            kwargs["eval_strategy"] = "epoch"
        # DeBERTa is numerically unstable with fp16; prefer bf16 on Ampere+.
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                kwargs["bf16"] = True
            elif "fp16" in arg_names:
                kwargs["fp16"] = True
        kwargs["max_grad_norm"] = 0.5

        training_args = TrainingArguments(**kwargs)
        trainer = WeightedLossTrainer(
            class_weights=weight_tensor,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_trainer_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        metric_value = float(eval_metrics.get("eval_malicious_precision", 0.0))
        if metric_value > best_metric:
            best_metric = metric_value
            best_lr = float(lr)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Transformer training did not produce a valid state.")

    for run_dir in run_dirs:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        selected_model_name,
        num_labels=len(LABEL_ORDER),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    final_model.load_state_dict(best_state)

    tokenizer.save_pretrained(str(cfg.model_dir))
    final_model.save_pretrained(str(cfg.model_dir))

    train_probs = _predict_probabilities(final_model, tokenizer, train_texts, cfg.max_length, cfg.eval_batch_size)
    val_probs = _predict_probabilities(final_model, tokenizer, val_texts, cfg.max_length, cfg.eval_batch_size)
    test_probs = _predict_probabilities(final_model, tokenizer, test_texts, cfg.max_length, cfg.eval_batch_size)

    train_pred = np.argmax(train_probs, axis=1)
    val_pred = np.argmax(val_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)
    train_metrics = _compute_metrics(y_train, train_pred)
    val_metrics = _compute_metrics(y_val, val_pred)
    test_metrics = _compute_metrics(y_test, test_pred)

    probability_scores = pd.concat(
        [
            _build_scores_table(train_df, train_probs),
            _build_scores_table(val_df, val_probs),
            _build_scores_table(test_df, test_probs),
        ],
        axis=0,
        ignore_index=True,
    )
    probability_scores_path = cfg.output_dir / "transformer_probability_scores.csv"
    probability_scores.to_csv(probability_scores_path, index=False)

    report = {
        "class_order": LABEL_ORDER,
        "label_to_id": label_to_id,
        "selected_backbone": selected_model_name,
        "best_learning_rate": best_lr,
        "target_metric": "malicious_precision",
        "class_weights": {
            "normal": float(cfg.class_weights[0]),
            "suspicious": float(cfg.class_weights[1]),
            "malicious": float(cfg.class_weights[2]),
        },
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test_ood": int(len(test_df)),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_ood_metrics": test_metrics,
        "artifact_paths": {
            "model_dir": str(cfg.model_dir.resolve()),
            "probability_scores": str(probability_scores_path.resolve()),
        },
        "config": {
            **asdict(cfg),
            "data_dir": str(cfg.data_dir),
            "model_dir": str(cfg.model_dir),
            "output_dir": str(cfg.output_dir),
            "model_candidates": list(cfg.model_candidates),
            "learning_rates": list(cfg.learning_rates),
            "class_weights": list(cfg.class_weights),
        },
    }
    report_path = cfg.model_dir / "training_report.json"
    report["artifact_paths"]["training_report"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
