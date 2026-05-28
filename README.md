# UPI Fraud Platform (Unified)

This repository unifies two complementary fraud systems into one production-style project:

- **Interaction Model** (`src/interaction`): behavioral/session fraud detection from text, URL, QR, device, and interaction signals.
- **Transaction Model** (`src/transaction`): tabular transaction fraud detection from amount, time, bank, network, and engineered risk signals.
- **Unified Fusion Model** (`src/unified`): parallel risk fusion that combines interaction + transaction outputs into one final risk score.

## Architecture

```text
main.py                                # Single orchestration CLI
src/interaction/                       # Existing hybrid interaction pipeline
src/transaction/                       # Classical ML transaction pipeline + SHAP
src/unified/                           # Fusion configuration, inference, evaluation
fraud_system/                          # Interaction runtime detector internals

interaction_data/                      # Interaction datasets
transaction_data/                      # Transaction datasets

models/interaction/                    # Interaction artifacts
models/transaction/                    # Transaction artifacts
models/unified/fusion_config.json      # Fusion policy

outputs/interaction/                   # Interaction evaluation outputs
outputs/transaction/                   # Transaction evaluation outputs
outputs/unified/                       # Unified evaluation outputs + graphs

docs/transaction/                      # Transaction model plots and SHAP assets
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Train

Train interaction model:

```powershell
python main.py train interaction --quick
```

Train transaction model:

```powershell
python main.py train transaction --data-path transaction_data/upi_transactions_2024.csv --with-shap
```

Configure unified fusion weights:

```powershell
python main.py train unified --interaction-weight 0.55 --transaction-weight 0.45 --high-threshold 0.70 --medium-threshold 0.40
```

### Inference

Interaction inference (single):

```powershell
python main.py infer interaction --input-text "Urgent UPI verification needed now" --url "http://secure-kyc-update.in/verify-now?ref=upi"
```

Transaction inference (batch):

```powershell
python main.py infer transaction --input-file transaction_data/upi_transactions_2024.csv --output-path outputs/transaction/predictions.csv
```

Unified inference (batch, fused output):

```powershell
python main.py infer unified --input-file transaction_data/upi_transactions_2024.csv --output-path outputs/unified/predictions.csv
```

### Evaluation

Interaction evaluation:

```powershell
python main.py evaluate interaction --split test_ood --with-xai
```

Transaction evaluation:

```powershell
python main.py evaluate transaction --data-path transaction_data/upi_transactions_2024.csv
```

Unified evaluation:

```powershell
python main.py evaluate unified --input-path transaction_data/upi_transactions_2024.csv --positive-label-threshold 0.5
```

Unified evaluation produces:

- `outputs/unified/predictions.csv`
- `outputs/unified/metrics.json`
- `outputs/unified/evaluation_graphs/*.png`

## Public Unified Inference Output

Unified inference returns:

- `interaction_score`
- `transaction_score`
- `fused_risk_score`
- `fused_risk_level`
- `per_model_explanations`

## Reproducible Training Notes

- Use fixed random seed (`--seed 42`) across training commands.
- Keep training and evaluation artifacts versioned under `models/*` and `outputs/*`.
- Update `models/unified/fusion_config.json` only via `python main.py train unified ...` for auditability.

## Model Card Summary

### Interaction Model
- **Type**: Transformer + XGBoost + IsolationForest hybrid ensemble.
- **Strength**: Rich behavior/context modeling and explainability artifacts.
- **Primary Artifacts**: `models/interaction/*`.

### Transaction Model
- **Type**: Logistic + RandomForest + XGBoost benchmark, risk-threshold tuned.
- **Strength**: Strong tabular signal engineering and SHAP transparency.
- **Primary Artifacts**: `models/transaction/*`, `docs/transaction/*`.

### Unified Fusion
- **Type**: Weighted parallel score fusion.
- **Strength**: Aggregates complementary model evidence into one operational risk decision.
- **Primary Artifacts**: `models/unified/fusion_config.json`, `outputs/unified/*`.

## Backward Compatibility

Legacy interaction commands are still available:

- `python main.py train-full-pipeline ...`
- `python main.py train-transformer ...`
- `python main.py build-feature-matrices ...`
- `python main.py train-xgboost ...`
- `python main.py train-isolation-forest ...`
- `python main.py train-ensemble ...`
- `python main.py finalize-models ...`
- `python main.py plot-evaluation ...`
