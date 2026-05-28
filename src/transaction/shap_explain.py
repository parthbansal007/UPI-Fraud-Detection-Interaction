""""

Generates:
  docs/shap_summary.png        — beeswarm summary of all features
  docs/shap_bar.png            — mean |SHAP| feature importance
  docs/shap_waterfall_fraud.png — waterfall for one fraudulent transaction
  docs/shap_waterfall_legit.png — waterfall for one legitimate transaction
  docs/shap_dependence_top1.png — dependence plot for the top feature
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from src.transaction.preprocess import preprocess_pipeline
from src.transaction.train import (
    TARGET_COLS, OHE_COLS, SMOOTH_FACTOR, RANDOM_STATE, TEST_SIZE,
    VAL_SIZE,
)

warnings.filterwarnings("ignore")


def _prepare_test_set(path: str, model_dir: str = "models/transaction"):
    """Re-creates the exact test set used during training (same seed)."""
    X, y = preprocess_pipeline(path)

    scaler         = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_names  = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    iso            = joblib.load(os.path.join(model_dir, "iso_forest.pkl"))

    # Replicate the train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    # Target encoding — must happen BEFORE OHE (same order as train.py)
    global_mean = float(y_tr.mean())
    for col in TARGET_COLS:
        if col not in X_tr.columns:
            continue
        agg = y_tr.groupby(X_tr[col]).agg(["count", "mean"])
        smoothed = (
            (agg["count"] * agg["mean"] + SMOOTH_FACTOR * global_mean)
            / (agg["count"] + SMOOTH_FACTOR)
        )
        X_tr[f"{col}_risk"]   = X_tr[col].map(smoothed).fillna(global_mean)
        X_test[f"{col}_risk"] = X_test[col].map(smoothed).fillna(global_mean)

    # OHE
    existing_ohe = [c for c in OHE_COLS if c in X_tr.columns]
    X_tr_ohe   = pd.get_dummies(X_tr,   columns=existing_ohe, drop_first=True)
    X_test_ohe = pd.get_dummies(X_test, columns=existing_ohe, drop_first=True)
    X_tr_ohe, X_test_ohe = X_tr_ohe.align(X_test_ohe, join="left", axis=1, fill_value=0)

    X_test_ohe = X_test_ohe.select_dtypes(exclude=["object"])
    # Align to exactly the features seen during training
    missing = [f for f in feature_names if f not in X_test_ohe.columns]
    for f in missing:
        X_test_ohe[f] = 0.0
    X_test_sc  = scaler.transform(X_test_ohe[feature_names])

    iso_scores = (-iso.decision_function(X_test_sc)).reshape(-1, 1)
    X_test_sc  = np.hstack([X_test_sc, iso_scores])

    feature_names_ext = feature_names + ["iso_anomaly_score"]
    return X_test_sc, y_test, feature_names_ext


def run_shap(path: str, model_name: str = "xgboost",
             model_dir: str = "models/transaction", docs_dir: str = "docs/transaction") -> None:

    os.makedirs(docs_dir, exist_ok=True)

    print(f"\nSHAP Explainability for {model_name.upper()}:")

    model         = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))
    X_test_sc, y_test, feature_names = _prepare_test_set(path, model_dir)

    print("  Computing SHAP values ...")
    if model_name in ("xgboost", "lightgbm", "rf"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sc)
        # LightGBM returns list [neg_class, pos_class] for binary
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        explainer   = shap.LinearExplainer(model, X_test_sc)
        shap_values = explainer.shap_values(X_test_sc)
        expected_value = explainer.expected_value

    #  1. SUMMARY 
    print("  1. Generating SHAP Summary Plot ...")
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values, X_test_sc,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title(f"SHAP Summary — {model_name.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "shap_summary.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 2. BAR PLOT 
    print("  2. Generating Feature Importance Bar Plot ...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test_sc,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title(f"Mean |SHAP| Importance — {model_name.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "shap_bar.png"), dpi=130, bbox_inches="tight")
    plt.close()

    #  3. TOP FEATURES 
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1]
    print("\n  TOP 10 FRAUD INDICATORS (by mean |SHAP|):")
    for rank, i in enumerate(top_idx[:10], 1):
        print(f"    {rank:>2}. {feature_names[i]:<40} {mean_abs_shap[i]:.5f}")

    #  4. WATERFALL — one FRAUD transaction 
    print("\n  3. Waterfall plot for a fraudulent transaction ...")
    y_test_arr   = np.asarray(y_test)
    fraud_indices = np.where(y_test_arr == 1)[0]

    if len(fraud_indices) > 0:
        idx = fraud_indices[0]
        expl_obj = shap.Explanation(
            values        = shap_values[idx],
            base_values   = expected_value,
            data          = X_test_sc[idx],
            feature_names = feature_names,
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(expl_obj, max_display=15, show=False)
        plt.title("SHAP Waterfall — Fraudulent Transaction", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(docs_dir, "shap_waterfall_fraud.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()

    # 5. WATERFALL — one LEGIT transaction 
    print("  4. Waterfall plot for a legitimate transaction ...")
    legit_indices = np.where(y_test_arr == 0)[0]
    if len(legit_indices) > 0:
        idx = legit_indices[0]
        expl_obj = shap.Explanation(
            values        = shap_values[idx],
            base_values   = expected_value,
            data          = X_test_sc[idx],
            feature_names = feature_names,
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(expl_obj, max_display=15, show=False)
        plt.title("SHAP Waterfall — Legitimate Transaction", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(docs_dir, "shap_waterfall_legit.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()

    #  6. DEPENDENCE PLOT — top feature 
    print("  5. Dependence plot for top feature ...")
    top_feature = feature_names[top_idx[0]]
    plt.figure(figsize=(9, 6))
    shap.dependence_plot(
        top_idx[0], shap_values, X_test_sc,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence — {top_feature}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "shap_dependence_top1.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n  All SHAP plots saved to '{docs_dir}/'")
