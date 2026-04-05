# UPI Interaction Fraud Detection

This repository now contains only the interaction fraud detection module and its artifacts.

## Folder Structure

```text
UPI-FRAUD-DETECTION/
|
|-- interaction_data/
|   |-- dataset_v2.csv
|   `-- dataset_encoded_v2.csv
|
|-- src/
|   `-- interaction/
|       |-- preprocess.py
|       |-- feature_engineering.py
|       |-- train_model.py
|       |-- evaluate.py
|       `-- xai.py
|
|-- models/
|   |-- interaction/
|   |-- xgboost_model.pkl
|   `-- scaler.pkl
|
|-- outputs/
|   |-- metrics.json
|   |-- predictions.csv
|   |-- shap_summary.png
|   `-- shap_force_plot.html
|
|-- notebooks/
|-- fraud_system/
|-- requirements.txt
|-- main.py
`-- train_fraud_system.py
```

## Commands

Train:

```bash
python main.py train --data-dir interaction_data --model-dir models/interaction --output-dir outputs
```

Quick train:

```bash
python main.py train --quick --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

Evaluate:

```bash
python main.py evaluate --split test_ood --with-xai
```

Single inference:

```bash
python main.py infer \
  --input-text "Urgent UPI verification needed now" \
  --url "http://secure-kyc-update.in/verify-now?ref=upi"
```

Backward-compatible shim:

```bash
python train_fraud_system.py --quick
```

## Public Interaction API

From `src.interaction`:

- `train_interaction_model(config: dict) -> dict`
- `load_interaction_model(model_dir: str) -> object`
- `predict_interaction_risk(df_or_inputs, model_dir: str) -> pandas.DataFrame`
- `detect_fraud(input_text, url, qr_data, device_info, model_dir="models/interaction") -> dict`
