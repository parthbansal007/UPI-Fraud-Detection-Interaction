from __future__ import annotations

import argparse
import json

from inference import predict_from_json
from model_training import load_encoded_dataset, train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UPI fraud detection intelligence system")
    parser.add_argument("--data-path", type=str, default="dataset_encoded_v2.csv", help="Path to the encoded CSV dataset")
    parser.add_argument("--audit-data-path", type=str, default="dataset_v2.csv", help="Path to the raw audit CSV dataset")
    parser.add_argument("--train", action="store_true", help="Train the fraud detection system")
    parser.add_argument("--evaluate", action="store_true", help="Train and emit the final test_ood report")
    parser.add_argument("--predict-json", type=str, help="Path to inference payload JSON")
    parser.add_argument("--model-path", type=str, default="trained_model.pkl", help="Saved model path")
    parser.add_argument("--feature-columns-path", type=str, default="feature_columns.pkl", help="Saved feature list path")
    parser.add_argument("--shap-explainer-path", type=str, default="shap_explainer.pkl", help="Saved SHAP explainer path")
    parser.add_argument("--predictions-output-path", type=str, default="predictions_with_scores.csv", help="Output CSV for scored predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train or args.evaluate:
        dataset = load_encoded_dataset(args.data_path, audit_dataset_path=args.audit_data_path).frame
        print(
            json.dumps(
                {
                    "rows": int(len(dataset)),
                    "labels": {key: int(value) for key, value in dataset["label"].value_counts().sort_index().items()},
                    "splits": {key: int(value) for key, value in dataset["split"].value_counts().to_dict().items()},
                },
                indent=2,
            )
        )
        report = train_pipeline(
            dataset_path=args.data_path,
            audit_dataset_path=args.audit_data_path,
            model_output_path=args.model_path,
            feature_output_path=args.feature_columns_path,
            shap_output_path=args.shap_explainer_path,
            predictions_output_path=args.predictions_output_path,
        )
        print(json.dumps(report, indent=2))
    if args.predict_json:
        prediction = predict_from_json(
            payload_path=args.predict_json,
            model_path=args.model_path,
            feature_columns_path=args.feature_columns_path,
            shap_explainer_path=args.shap_explainer_path,
        )
        print(json.dumps(prediction, indent=2))
    if not any([args.train, args.evaluate, args.predict_json]):
        raise SystemExit("No action selected. Use --train, --evaluate, or --predict-json.")


if __name__ == "__main__":
    main()
