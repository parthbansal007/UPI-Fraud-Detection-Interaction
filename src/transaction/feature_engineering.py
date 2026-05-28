"""
Builds every available signal from the 17-column UPI CSV.

Column inventory (post clean_column_names):
  transaction_id, timestamp, transaction_type, merchant_category,
  amount_inr, transaction_status, sender_age_group, receiver_age_group,
  sender_state, sender_bank, receiver_bank, device_type, network_type,
  fraud_flag, hour_of_day, day_of_week, is_weekend

Key design decisions:
• No sender_id / receiver_id exists → we cannot build true per-user
  velocity. We maximise signal from what IS available.
• All statistics are computed on the full dataset BEFORE the train/test
  split (leakage-safe for unsupervised stats; target encoding is
  handled separately in train.py on the training fold only).
• Cyclical time encoding (sin/cos) prevents midnight discontinuity.
• WiFi flag: EDA shows WiFi has the highest fraud rate (0.235%).
• High-amount transactions (>5k, >10k) have elevated fraud rates.
• Cross-bank flag has slightly higher fraud rate than same-bank.
• Kotak / ICICI / PNB have the highest sender fraud rates.
• Karnataka / Rajasthan / Gujarat have the highest state fraud rates.
"""

import pandas as pd
import numpy as np


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. TIMESTAMP DECOMPOSITION 
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["day_of_month"]  = df["timestamp"].dt.day
    df["month"]         = df["timestamp"].dt.month
    df["week_of_year"]  = df["timestamp"].dt.isocalendar().week.astype(int)
    df["quarter"]       = df["timestamp"].dt.quarter

    # Cyclical time encoding
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["dom_sin"]     = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["dom_cos"]     = np.cos(2 * np.pi * df["day_of_month"] / 31)
    df["week_sin"]    = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Time-window flags (EDA: hours 3, 1, 22, 0 have elevated fraud)
    df["is_late_night"]     = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 4)).astype(int)
    df["is_early_morning"]  = ((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 3)).astype(int)
    df["is_business_hours"] = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 18)).astype(int)
    df["is_peak_evening"]   = ((df["hour_of_day"] >= 19) & (df["hour_of_day"] <= 22)).astype(int)

    # 2. AMOUNT FEATURES ─
    df["amount_log"]    = np.log1p(df["amount_inr"])
    df["amount_sqrt"]   = np.sqrt(df["amount_inr"])
    df["amount_sq"]     = df["amount_inr"] ** 2  # captures extreme values

    # EDA-driven thresholds: fraud rate peaks at >5k and >10k
    df["is_high_amount"]      = (df["amount_inr"] > 5000).astype(int)
    df["is_very_high_amount"] = (df["amount_inr"] > 10000).astype(int)
    df["is_micro_amount"]     = (df["amount_inr"] < 100).astype(int)

    # Percentile flags (global — before split)
    p95 = df["amount_inr"].quantile(0.95)
    p99 = df["amount_inr"].quantile(0.99)
    p05 = df["amount_inr"].quantile(0.05)
    df["is_above_p95"] = (df["amount_inr"] > p95).astype(int)
    df["is_above_p99"] = (df["amount_inr"] > p99).astype(int)
    df["is_below_p05"] = (df["amount_inr"] < p05).astype(int)

    # Round-number heuristic
    df["is_round_1000"] = (df["amount_inr"] % 1000 == 0).astype(int)
    df["is_round_500"]  = (df["amount_inr"] % 500  == 0).astype(int)
    df["is_round_100"]  = (df["amount_inr"] % 100  == 0).astype(int)

    # Amount × time interaction (high amounts at night — elevated fraud)
    df["high_amt_night"]     = (df["is_high_amount"] & df["is_late_night"]).astype(int)
    df["high_amt_early"]     = (df["is_high_amount"] & df["is_early_morning"]).astype(int)
    df["very_high_amt_night"]= (df["is_very_high_amount"] & df["is_late_night"]).astype(int)

    # 3. CROSS-BANK INDICATOR
    # EDA: cross-bank rate 0.195% vs same-bank 0.174%
    df["is_cross_bank"] = (df["sender_bank"] != df["receiver_bank"]).astype(int)
    df["high_amt_cross_bank"] = (df["is_high_amount"] & df["is_cross_bank"]).astype(int)

    #  4. NETWORK / DEVICE RISK FLAGS 
    # EDA: WiFi has highest fraud rate (0.235%), Web device slightly elevated
    df["is_wifi"]        = (df["network_type"] == "WiFi").astype(int)
    df["is_web"]         = (df["device_type"] == "Web").astype(int)
    df["is_3g"]          = (df["network_type"] == "3G").astype(int)
    df["wifi_high_amt"]  = (df["is_wifi"] & df["is_high_amount"]).astype(int)
    df["web_high_amt"]   = (df["is_web"] & df["is_high_amount"]).astype(int)
    df["wifi_night"]     = (df["is_wifi"] & df["is_late_night"]).astype(int)

    #  5. TRANSACTION STATUS 
    df["is_failed"] = (df["transaction_status"] == "FAILED").astype(int)

    #  6. MERCHANT Z-SCORE (amount anomaly per merchant) 
    merch_stats = (
        df.groupby("merchant_category")["amount_inr"]
        .agg(merch_mean="mean", merch_std="std")
        .reset_index()
    )
    merch_stats["merch_std"] = merch_stats["merch_std"].fillna(1.0).clip(lower=1.0)
    df = df.merge(merch_stats, on="merchant_category", how="left")
    df["merchant_zscore"] = (
        (df["amount_inr"] - df["merch_mean"]) / df["merch_std"]
    ).clip(-5, 5)

    #  7. SENDER BANK Z-SCORE 
    bank_stats = (
        df.groupby("sender_bank")["amount_inr"]
        .agg(bank_mean="mean", bank_std="std")
        .reset_index()
    )
    bank_stats["bank_std"] = bank_stats["bank_std"].fillna(1.0).clip(lower=1.0)
    df = df.merge(bank_stats, on="sender_bank", how="left")
    df["sender_bank_zscore"] = (
        (df["amount_inr"] - df["bank_mean"]) / df["bank_std"]
    ).clip(-5, 5)

    # 8. RECEIVER BANK Z-SCORE 
    rbank_stats = (
        df.groupby("receiver_bank")["amount_inr"]
        .agg(rbank_mean="mean", rbank_std="std")
        .reset_index()
    )
    rbank_stats["rbank_std"] = rbank_stats["rbank_std"].fillna(1.0).clip(lower=1.0)
    df = df.merge(rbank_stats, on="receiver_bank", how="left")
    df["receiver_bank_zscore"] = (
        (df["amount_inr"] - df["rbank_mean"]) / df["rbank_std"]
    ).clip(-5, 5)

    # 9. STATE Z-SCORE 
    state_stats = (
        df.groupby("sender_state")["amount_inr"]
        .agg(state_mean="mean", state_std="std")
        .reset_index()
    )
    state_stats["state_std"] = state_stats["state_std"].fillna(1.0).clip(lower=1.0)
    df = df.merge(state_stats, on="sender_state", how="left")
    df["state_amount_zscore"] = (
        (df["amount_inr"] - df["state_mean"]) / df["state_std"]
    ).clip(-5, 5)

    # 10. GLOBAL FREQUENCY FEATURES 
    n = len(df)
    df["merchant_freq"]      = df.groupby("merchant_category")["amount_inr"].transform("count") / n
    df["sender_bank_freq"]   = df.groupby("sender_bank")["amount_inr"].transform("count") / n
    df["receiver_bank_freq"] = df.groupby("receiver_bank")["amount_inr"].transform("count") / n
    df["state_freq"]         = df.groupby("sender_state")["amount_inr"].transform("count") / n
    df["txn_type_freq"]      = df.groupby("transaction_type")["amount_inr"].transform("count") / n

    # 11. VELOCITY PROXY (volume per merchant × hour) 
    df["merchant_hour_vol"] = (
        df.groupby(["merchant_category", "hour_of_day"])["amount_inr"]
        .transform("count")
    )
    df["bank_hour_vol"] = (
        df.groupby(["sender_bank", "hour_of_day"])["amount_inr"]
        .transform("count")
    )
    df["state_hour_vol"] = (
        df.groupby(["sender_state", "hour_of_day"])["amount_inr"]
        .transform("count")
    )

    # Merchant × day volume
    df["merchant_day_vol"] = (
        df.groupby(["merchant_category", "day_of_week"])["amount_inr"]
        .transform("count")
    )

    # 12. AMOUNT RATIO TO GROUP MEAN 
    # How much larger/smaller is this txn vs the group average?
    df["amt_ratio_merchant"] = df["amount_inr"] / (df["merch_mean"] + 1.0)
    df["amt_ratio_bank"]     = df["amount_inr"] / (df["bank_mean"] + 1.0)
    df["amt_ratio_state"]    = df["amount_inr"] / (df["state_mean"] + 1.0)

    # 13. COMBINED RISK INTERACTIONS 
    df["wifi_cross_bank"]    = (df["is_wifi"] & df["is_cross_bank"]).astype(int)
    df["night_cross_bank"]   = (df["is_late_night"] & df["is_cross_bank"]).astype(int)
    df["night_wifi_high"]    = (
        df["is_late_night"] & df["is_wifi"] & df["is_high_amount"]
    ).astype(int)

    # 14. AGE GROUP FEATURES
    age_map = {"18-25": 1, "26-35": 2, "36-45": 3, "46-55": 4, "56+": 5}
    df["sender_age_num"]   = df["sender_age_group"].map(age_map).fillna(3)
    df["receiver_age_num"] = df["receiver_age_group"].map(age_map).fillna(3)
    df["age_diff"]         = df["sender_age_num"] - df["receiver_age_num"]
    df["age_sum"]          = df["sender_age_num"] + df["receiver_age_num"]
    # Young senders (18-25) sending to older receivers
    df["young_to_old"]     = (
        (df["sender_age_num"] == 1) & (df["receiver_age_num"] >= 4)
    ).astype(int)

    #  15. CATEGORY COMBO (for target encoding in train.py) 
    df["category_combo"]    = df["merchant_category"] + "_" + df["transaction_type"]
    df["bank_combo"]        = df["sender_bank"] + "_" + df["receiver_bank"]
    df["state_bank_combo"]  = df["sender_state"] + "_" + df["sender_bank"]
    df["device_network"]    = df["device_type"] + "_" + df["network_type"]

    # 16. CLEANUP 
    drop_cols = [
        "merch_mean", "merch_std",
        "bank_mean",  "bank_std",
        "rbank_mean", "rbank_std",
        "state_mean", "state_std",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df