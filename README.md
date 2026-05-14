# UPI Interaction Fraud Detection

This project predicts interaction-level UPI fraud risk from message text, URLs, QR payloads, device state, and behavioral event data.

The finalized runtime artifacts live in `models/interaction`, and the main command-line entry point is `main.py`.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the project virtual environment when available:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

## Single Inference

```powershell
.\.venv\Scripts\python.exe main.py infer `
  --input-text "Urgent UPI verification needed now" `
  --url "http://secure-kyc-update.in/verify-now?ref=upi"
```

The response contains:

- `predicted_label`: `normal`, `suspicious`, or `malicious`
- `risk_level`: `LOW`, `MEDIUM`, or `HIGH`
- `fraud_probability`
- per-class probabilities
- short feature explanations

## Batch Inference

The batch path supports normal online-style columns such as `input_text`, `url`, and `qr_data`.

It also supports the supplied raw interaction dataset shape directly:

```powershell
.\.venv\Scripts\python.exe main.py infer `
  --input-file interaction_data\dataset_v2.csv `
  --output-path outputs\predictions.csv
```

Batch outputs include:

- `session_id`
- `interaction_label`
- `interaction_risk_score`
- `interaction_risk_level`
- `normal_probability`
- `suspicious_probability`
- `malicious_probability`
- `true_label` when available
- `source_split`
- `scenario_family`

## Evaluate

```powershell
.\.venv\Scripts\python.exe main.py evaluate --split test_ood
```

Latest finalized `test_ood` evaluation:

```text
accuracy: 0.6940
malicious_precision: 0.7178
malicious_recall: 0.5847
malicious_f1: 0.6444
roc_auc: 0.8965
```

Outputs are written to:

```text
outputs/metrics.json
outputs/predictions.csv
```

## Evaluation Graphs

Generate the core model-evaluation graph bundle:

```powershell
.\.venv\Scripts\python.exe main.py plot-evaluation `
  --predictions-path outputs\predictions.csv `
  --output-dir outputs\evaluation_graphs
```

Graphs include confusion matrices, ROC curves, precision-recall curves, class metric bars, malicious threshold tradeoffs, score distributions, calibration, and confidence distribution.

## Final Reports

Important project reports:

```text
models/interaction/final_project_report.json
models/interaction/runtime_calibration_report.json
models/interaction/evaluation_report.json
models/interaction/ensemble_report.json
models/interaction/xgboost_tuning_report.json
models/interaction/transformer/training_report.json
```

The finalized runtime uses calibrated thresholds selected on the validation split:

```text
ensemble_weights: transformer=0.35, xgboost=0.55, anomaly=0.10
malicious_threshold: 0.34
suspicious_threshold: 0.35
```

## Training

Quick smoke training:

```powershell
.\.venv\Scripts\python.exe main.py train --quick --epochs 1 --train-batch-size 8 --eval-batch-size 16
```

Full train/finalize/evaluate path:

```powershell
.\.venv\Scripts\python.exe main.py train-full-pipeline `
  --data-dir interaction_data `
  --model-dir models\interaction `
  --output-dir outputs
```

Backward-compatible shim:

```powershell
.\.venv\Scripts\python.exe train_fraud_system.py --quick
```

## Public Python API

```python
from src.interaction import detect_fraud, predict_interaction_risk

result = detect_fraud(
    input_text="Urgent UPI verification needed now",
    url="http://secure-kyc-update.in/verify-now?ref=upi",
    qr_data="",
)
```

Available API functions:

- `train_interaction_model(config: dict) -> dict`
- `load_interaction_model(model_dir: str) -> object`
- `predict_interaction_risk(df_or_inputs, model_dir: str) -> pandas.DataFrame`
- `detect_fraud(input_text, url, qr_data, device_info, model_dir="models/interaction") -> dict`
