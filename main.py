from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.interaction import (
    detect_fraud,
    evaluate_fraud_detection_system,
    evaluate_interaction_model,
    export_evaluation_graphs,
    finalize_trained_fraud_models,
    generate_transformer_feature_matrices,
    predict_interaction_risk,
    train_ensemble_model,
    train_interaction_model,
    train_isolation_forest_model,
    train_optimized_xgboost,
    train_transformer_text_model,
)
from src.transaction import evaluate_transaction_model, predict_transaction_risk, train_transaction_model
from src.unified import evaluate_unified_model, infer_unified_risk, train_unified_fusion


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UPI fraud platform orchestrator (interaction + transaction + unified fusion).")
    root_subparsers = parser.add_subparsers(dest="action", required=True)

    train_parser = root_subparsers.add_parser("train", help="Train interaction, transaction, or unified fusion configuration")
    train_parser.add_argument("domain", nargs="?", choices=["interaction", "transaction", "unified"], default="interaction")
    train_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    train_parser.add_argument("--data-path", type=Path, default=Path("transaction_data/upi_transactions_2024.csv"))
    train_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction"))
    train_parser.add_argument("--transaction-model-dir", type=Path, default=Path("models/transaction"))
    train_parser.add_argument("--docs-dir", type=Path, default=Path("docs/transaction"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/interaction"))
    train_parser.add_argument("--transaction-output-dir", type=Path, default=Path("outputs/transaction"))
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--quick", action="store_true")
    train_parser.add_argument("--max-length", type=int, default=96)
    train_parser.add_argument("--epochs", type=float, default=4.0)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--with-shap", action="store_true")
    train_parser.add_argument("--model-for-shap", type=str, default="xgboost")
    train_parser.add_argument("--interaction-weight", type=float, default=0.55)
    train_parser.add_argument("--transaction-weight", type=float, default=0.45)
    train_parser.add_argument("--high-threshold", type=float, default=0.70)
    train_parser.add_argument("--medium-threshold", type=float, default=0.40)

    infer_parser = root_subparsers.add_parser("infer", help="Run inference for interaction, transaction, or unified fusion")
    infer_parser.add_argument("domain", nargs="?", choices=["interaction", "transaction", "unified"], default="interaction")
    infer_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction"))
    infer_parser.add_argument("--transaction-model-dir", type=Path, default=Path("models/transaction"))
    infer_parser.add_argument("--input-file", type=Path, default=None, help="CSV for batch inference")
    infer_parser.add_argument("--output-path", type=Path, default=Path("outputs/predictions.csv"))
    infer_parser.add_argument("--input-text", type=str, default="")
    infer_parser.add_argument("--url", type=str, default="https://unknown.local")
    infer_parser.add_argument("--qr-data", type=str, default="")
    infer_parser.add_argument("--device-info-json", type=str, default="")
    infer_parser.add_argument("--record-json", type=str, default="", help="JSON record for transaction/unified single inference")

    evaluate_parser = root_subparsers.add_parser("evaluate", help="Evaluate interaction, transaction, or unified fusion")
    evaluate_parser.add_argument("domain", nargs="?", choices=["interaction", "transaction", "unified"], default="interaction")
    evaluate_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    evaluate_parser.add_argument("--data-path", type=Path, default=Path("transaction_data/upi_transactions_2024.csv"))
    evaluate_parser.add_argument("--input-path", type=Path, default=Path("transaction_data/upi_transactions_2024.csv"))
    evaluate_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction"))
    evaluate_parser.add_argument("--transaction-model-dir", type=Path, default=Path("models/transaction"))
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    evaluate_parser.add_argument("--split", type=str, default="test_ood")
    evaluate_parser.add_argument("--with-xai", action="store_true")
    evaluate_parser.add_argument("--xai-sample-size", type=int, default=200)
    evaluate_parser.add_argument("--positive-label-threshold", type=float, default=0.5)

    # Legacy interaction commands retained for compatibility.
    full_parser = root_subparsers.add_parser(
        "train-full-pipeline",
        help="Train, finalize, and evaluate an inference-ready interaction model",
    )
    full_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    full_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction"))
    full_parser.add_argument("--output-dir", type=Path, default=Path("outputs/interaction"))
    full_parser.add_argument("--seed", type=int, default=42)
    full_parser.add_argument("--quick", action="store_true")
    full_parser.add_argument("--max-length", type=int, default=96)
    full_parser.add_argument("--epochs", type=float, default=4.0)
    full_parser.add_argument("--train-batch-size", type=int, default=16)
    full_parser.add_argument("--eval-batch-size", type=int, default=32)
    full_parser.add_argument("--with-xai", action="store_true")

    transformer_parser = root_subparsers.add_parser("train-transformer", help="Train transformer text model")
    transformer_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    transformer_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction/transformer"))
    transformer_parser.add_argument("--output-dir", type=Path, default=Path("outputs/interaction"))
    transformer_parser.add_argument("--seed", type=int, default=42)
    transformer_parser.add_argument("--model-name", type=str, default="distilroberta-base")
    transformer_parser.add_argument("--learning-rate", type=float, default=2e-5)
    transformer_parser.add_argument("--tokenization-batch-size", type=int, default=256)
    transformer_parser.add_argument("--early-stopping-patience", type=int, default=2)
    transformer_parser.add_argument("--max-length", type=int, default=96)
    transformer_parser.add_argument("--epochs", type=float, default=4.0)
    transformer_parser.add_argument("--train-batch-size", type=int, default=16)
    transformer_parser.add_argument("--eval-batch-size", type=int, default=32)

    matrix_parser = root_subparsers.add_parser(
        "build-feature-matrices",
        help="Generate transformer embeddings and save concatenated feature matrices",
    )
    matrix_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    matrix_parser.add_argument("--transformer-dir", type=Path, default=Path("models/interaction/transformer"))
    matrix_parser.add_argument("--output-dir", type=Path, default=Path("outputs/processed_matrices"))
    matrix_parser.add_argument("--seed", type=int, default=42)
    matrix_parser.add_argument("--max-length", type=int, default=None)
    matrix_parser.add_argument("--batch-size", type=int, default=32)

    xgb_parser = root_subparsers.add_parser("train-xgboost", help="Train tuned XGBoost on transformer+structured matrices")
    xgb_parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/processed_matrices"))
    xgb_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    xgb_parser.add_argument("--transformer-dir", type=Path, default=Path("models/interaction/transformer"))
    xgb_parser.add_argument("--model-path", type=Path, default=Path("models/interaction/xgboost_model.json"))
    xgb_parser.add_argument("--report-path", type=Path, default=Path("models/interaction/xgboost_tuning_report.json"))
    xgb_parser.add_argument("--seed", type=int, default=42)
    xgb_parser.add_argument("--early-stopping-rounds", type=int, default=50)
    xgb_parser.add_argument("--weight-normal", type=float, default=1.0)
    xgb_parser.add_argument("--weight-suspicious", type=float, default=2.0)
    xgb_parser.add_argument("--weight-malicious", type=float, default=4.0)

    iso_parser = root_subparsers.add_parser("train-isolation-forest", help="Train Isolation Forest anomaly model")
    iso_parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/processed_matrices"))
    iso_parser.add_argument("--data-dir", type=Path, default=Path("interaction_data"))
    iso_parser.add_argument("--transformer-dir", type=Path, default=Path("models/interaction/transformer"))
    iso_parser.add_argument("--model-path", type=Path, default=Path("models/interaction/isolation_forest.joblib"))
    iso_parser.add_argument("--scores-path", type=Path, default=Path("outputs/isolation_forest_scores.csv"))
    iso_parser.add_argument("--report-path", type=Path, default=Path("models/interaction/isolation_forest_report.json"))
    iso_parser.add_argument("--seed", type=int, default=42)
    iso_parser.add_argument("--n-estimators", type=int, default=300)
    iso_parser.add_argument("--contamination", type=float, default=0.18)
    iso_parser.add_argument("--normalize-low-quantile", type=float, default=0.05)
    iso_parser.add_argument("--normalize-high-quantile", type=float, default=0.95)

    ens_parser = root_subparsers.add_parser("train-ensemble", help="Train tuned weighted ensemble for fraud probability")
    ens_parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/processed_matrices"))
    ens_parser.add_argument("--transformer-scores-path", type=Path, default=Path("outputs/transformer_probability_scores.csv"))
    ens_parser.add_argument("--xgboost-model-path", type=Path, default=Path("models/interaction/xgboost_model.json"))
    ens_parser.add_argument("--anomaly-scores-path", type=Path, default=Path("outputs/isolation_forest_scores.csv"))
    ens_parser.add_argument("--output-scores-path", type=Path, default=Path("outputs/ensemble_fraud_probabilities.csv"))
    ens_parser.add_argument("--output-report-path", type=Path, default=Path("models/interaction/ensemble_report.json"))
    ens_parser.add_argument("--tune-step", type=float, default=0.05)
    ens_parser.add_argument("--decision-threshold", type=float, default=0.5)
    ens_parser.add_argument("--malicious-threshold-min", type=float, default=0.7)
    ens_parser.add_argument("--malicious-threshold-max", type=float, default=0.9)
    ens_parser.add_argument("--suspicious-threshold-min", type=float, default=0.4)
    ens_parser.add_argument("--suspicious-threshold-max", type=float, default=0.7)
    ens_parser.add_argument("--threshold-step", type=float, default=0.01)
    ens_parser.add_argument("--min-malicious-precision", type=float, default=0.7)

    eval_system_parser = root_subparsers.add_parser(
        "evaluate-system",
        help="Evaluate fraud detection system metrics on validation and test_ood splits",
    )
    eval_system_parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("outputs/ensemble_fraud_probabilities.csv"),
    )
    eval_system_parser.add_argument("--output-dir", type=Path, default=Path("outputs/interaction"))

    graph_parser = root_subparsers.add_parser(
        "plot-evaluation",
        help="Generate evaluation graphs from predictions.csv",
    )
    graph_parser.add_argument("--predictions-path", type=Path, default=Path("outputs/interaction/predictions.csv"))
    graph_parser.add_argument("--output-dir", type=Path, default=Path("outputs/interaction/evaluation_graphs"))
    graph_parser.add_argument("--metadata-path", type=Path, default=Path("models/interaction/metadata.json"))
    graph_parser.add_argument("--dpi", type=int, default=180)

    finalize_parser = root_subparsers.add_parser(
        "finalize-models",
        help="Finalize trained fraud models and write inference-compatible metadata",
    )
    finalize_parser.add_argument("--model-dir", type=Path, default=Path("models/interaction"))
    finalize_parser.add_argument("--metadata-path", type=Path, default=Path("models/interaction/metadata.json"))
    finalize_parser.add_argument("--ensemble-report-path", type=Path, default=Path("models/interaction/ensemble_report.json"))
    finalize_parser.add_argument(
        "--transformer-report-path",
        type=Path,
        default=Path("models/interaction/transformer/training_report.json"),
    )
    finalize_parser.add_argument("--xgboost-report-path", type=Path, default=Path("models/interaction/xgboost_tuning_report.json"))
    finalize_parser.add_argument(
        "--isolation-forest-report-path",
        type=Path,
        default=Path("models/interaction/isolation_forest_report.json"),
    )

    return parser


def _run_train(args: argparse.Namespace) -> None:
    if args.domain == "interaction":
        report = train_interaction_model(
            {
                "data_dir": args.data_dir,
                "model_dir": args.model_dir,
                "output_dir": args.output_dir,
                "seed": args.seed,
                "quick": args.quick,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
            }
        )
        print(json.dumps(report, indent=2))
        return

    if args.domain == "transaction":
        report = train_transaction_model(
            {
                "data_path": args.data_path,
                "model_dir": args.transaction_model_dir,
                "docs_dir": args.docs_dir,
                "output_dir": args.transaction_output_dir,
                "with_shap": args.with_shap,
                "model_for_shap": args.model_for_shap,
            }
        )
        print(json.dumps(report, indent=2))
        return

    report = train_unified_fusion(
        {
            "interaction_weight": args.interaction_weight,
            "transaction_weight": args.transaction_weight,
            "high_threshold": args.high_threshold,
            "medium_threshold": args.medium_threshold,
            "interaction_model_dir": str(args.model_dir),
            "transaction_model_dir": str(args.transaction_model_dir),
        }
    )
    print(json.dumps(report, indent=2))


def _run_infer(args: argparse.Namespace) -> None:
    if args.domain == "interaction":
        if args.input_file is not None:
            df = pd.read_csv(args.input_file)
            pred_df = predict_interaction_risk(df_or_inputs=df, model_dir=args.model_dir)
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(args.output_path, index=False)
            print(json.dumps({"rows": int(len(pred_df)), "output_path": str(args.output_path.resolve())}, indent=2))
            return

        result = detect_fraud(
            input_text=args.input_text,
            url=args.url,
            qr_data=args.qr_data,
            device_info=_json_arg(args.device_info_json),
            model_dir=args.model_dir,
        )
        print(json.dumps(result, indent=2))
        return

    if args.domain == "transaction":
        if args.input_file is not None:
            df = pd.read_csv(args.input_file)
        else:
            df = pd.DataFrame([_json_arg(args.record_json)]) if args.record_json else pd.DataFrame([{}])

        pred_df = predict_transaction_risk(df_or_inputs=df, model_dir=args.transaction_model_dir)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_path, index=False)
        print(json.dumps({"rows": int(len(pred_df)), "output_path": str(args.output_path.resolve())}, indent=2))
        return

    if args.input_file is not None:
        df = pd.read_csv(args.input_file)
    else:
        payload = _json_arg(args.record_json) if args.record_json else {}
        payload.setdefault("input_text", args.input_text)
        payload.setdefault("url", args.url)
        payload.setdefault("qr_data", args.qr_data)
        payload.setdefault("device_info", _json_arg(args.device_info_json))
        df = pd.DataFrame([payload])

    pred_df = infer_unified_risk(
        df_or_inputs=df,
        config={
            "interaction_model_dir": str(args.model_dir),
            "transaction_model_dir": str(args.transaction_model_dir),
        },
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_path, index=False)
    print(json.dumps({"rows": int(len(pred_df)), "output_path": str(args.output_path.resolve())}, indent=2))


def _run_evaluate(args: argparse.Namespace) -> None:
    if args.domain == "interaction":
        report = evaluate_interaction_model(
            {
                "data_dir": args.data_dir,
                "model_dir": args.model_dir,
                "output_dir": args.output_dir / "interaction",
                "split": args.split,
                "generate_xai": args.with_xai,
                "xai_sample_size": args.xai_sample_size,
            }
        )
        print(json.dumps(report, indent=2))
        return

    if args.domain == "transaction":
        report = evaluate_transaction_model(
            {
                "data_path": args.data_path,
                "model_dir": args.transaction_model_dir,
                "output_dir": args.output_dir / "transaction",
            }
        )
        print(json.dumps(report, indent=2))
        return

    report = evaluate_unified_model(
        {
            "input_path": args.input_path,
            "output_dir": args.output_dir / "unified",
            "positive_label_threshold": args.positive_label_threshold,
        }
    )
    print(json.dumps(report, indent=2))


def _run_full_pipeline(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {}
    report["training"] = train_interaction_model(
        {
            "data_dir": args.data_dir,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "quick": args.quick,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        }
    )
    try:
        report["finalization"] = finalize_trained_fraud_models(
            {
                "model_dir": args.model_dir,
                "metadata_path": args.model_dir / "metadata.json",
                "ensemble_report_path": args.model_dir / "ensemble_report.json",
                "transformer_report_path": args.model_dir / "transformer" / "training_report.json",
                "xgboost_report_path": args.model_dir / "xgboost_tuning_report.json",
                "isolation_forest_report_path": args.model_dir / "isolation_forest_report.json",
            }
        )
    except FileNotFoundError as exc:
        report["finalization"] = {
            "status": "skipped",
            "reason": str(exc),
        }
    report["evaluation"] = evaluate_interaction_model(
        {
            "data_dir": args.data_dir,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
            "split": "test_ood",
            "generate_xai": args.with_xai,
        }
    )
    print(json.dumps(report, indent=2))


def _run_transformer_train(args: argparse.Namespace) -> None:
    report = train_transformer_text_model(
        {
            "data_dir": args.data_dir,
            "model_dir": args.model_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "model_candidates": (args.model_name,),
            "learning_rates": (args.learning_rate,),
            "tokenization_batch_size": args.tokenization_batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
        }
    )
    print(json.dumps(report, indent=2))


def _run_feature_matrix_build(args: argparse.Namespace) -> None:
    report = generate_transformer_feature_matrices(
        {
            "data_dir": args.data_dir,
            "transformer_dir": args.transformer_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        }
    )
    print(json.dumps(report, indent=2))


def _run_xgboost_train(args: argparse.Namespace) -> None:
    report = train_optimized_xgboost(
        {
            "matrix_dir": args.matrix_dir,
            "data_dir": args.data_dir,
            "transformer_dir": args.transformer_dir,
            "output_model_path": args.model_path,
            "output_report_path": args.report_path,
            "random_state": args.seed,
            "early_stopping_rounds": args.early_stopping_rounds,
            "class_weights": (args.weight_normal, args.weight_suspicious, args.weight_malicious),
        }
    )
    print(json.dumps(report, indent=2))


def _run_isolation_forest_train(args: argparse.Namespace) -> None:
    report = train_isolation_forest_model(
        {
            "matrix_dir": args.matrix_dir,
            "data_dir": args.data_dir,
            "transformer_dir": args.transformer_dir,
            "output_model_path": args.model_path,
            "output_scores_path": args.scores_path,
            "output_report_path": args.report_path,
            "random_state": args.seed,
            "n_estimators": args.n_estimators,
            "contamination": args.contamination,
            "normalize_low_quantile": args.normalize_low_quantile,
            "normalize_high_quantile": args.normalize_high_quantile,
        }
    )
    print(json.dumps(report, indent=2))


def _run_ensemble_train(args: argparse.Namespace) -> None:
    report = train_ensemble_model(
        {
            "matrix_dir": args.matrix_dir,
            "transformer_scores_path": args.transformer_scores_path,
            "xgboost_model_path": args.xgboost_model_path,
            "anomaly_scores_path": args.anomaly_scores_path,
            "output_scores_path": args.output_scores_path,
            "output_report_path": args.output_report_path,
            "tune_step": args.tune_step,
            "decision_threshold": args.decision_threshold,
            "malicious_threshold_min": args.malicious_threshold_min,
            "malicious_threshold_max": args.malicious_threshold_max,
            "suspicious_threshold_min": args.suspicious_threshold_min,
            "suspicious_threshold_max": args.suspicious_threshold_max,
            "threshold_step": args.threshold_step,
            "min_malicious_precision": args.min_malicious_precision,
        }
    )
    print(json.dumps(report, indent=2))


def _run_system_evaluate(args: argparse.Namespace) -> None:
    report = evaluate_fraud_detection_system(
        {
            "predictions_path": args.predictions_path,
            "output_dir": args.output_dir,
        }
    )
    print(json.dumps(report, indent=2))


def _run_evaluation_graphs(args: argparse.Namespace) -> None:
    report = export_evaluation_graphs(
        {
            "predictions_path": args.predictions_path,
            "output_dir": args.output_dir,
            "metadata_path": args.metadata_path,
            "dpi": args.dpi,
        }
    )
    print(json.dumps(report, indent=2))


def _run_finalize_models(args: argparse.Namespace) -> None:
    report = finalize_trained_fraud_models(
        {
            "model_dir": args.model_dir,
            "metadata_path": args.metadata_path,
            "ensemble_report_path": args.ensemble_report_path,
            "transformer_report_path": args.transformer_report_path,
            "xgboost_report_path": args.xgboost_report_path,
            "isolation_forest_report_path": args.isolation_forest_report_path,
        }
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.action == "train":
        _run_train(args)
        return
    if args.action == "infer":
        _run_infer(args)
        return
    if args.action == "evaluate":
        _run_evaluate(args)
        return
    if args.action == "train-full-pipeline":
        _run_full_pipeline(args)
        return
    if args.action == "train-transformer":
        _run_transformer_train(args)
        return
    if args.action == "build-feature-matrices":
        _run_feature_matrix_build(args)
        return
    if args.action == "train-xgboost":
        _run_xgboost_train(args)
        return
    if args.action == "train-isolation-forest":
        _run_isolation_forest_train(args)
        return
    if args.action == "train-ensemble":
        _run_ensemble_train(args)
        return
    if args.action == "evaluate-system":
        _run_system_evaluate(args)
        return
    if args.action == "plot-evaluation":
        _run_evaluation_graphs(args)
        return
    if args.action == "finalize-models":
        _run_finalize_models(args)
        return

    parser.error("Unsupported command.")


if __name__ == "__main__":
    main()
