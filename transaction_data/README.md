# Transaction Dataset Placeholder

Place the transaction-level training/evaluation CSV here.

Expected file used by default commands:

- `transaction_data/upi_transactions_2024.csv`

Expected minimum columns:

- `timestamp`
- `transaction_type`
- `merchant_category`
- `amount_inr`
- `transaction_status`
- `sender_age_group`
- `receiver_age_group`
- `sender_state`
- `sender_bank`
- `receiver_bank`
- `device_type`
- `network_type`
- `fraud_flag` (required for evaluation)
