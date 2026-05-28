"""
UPI Fraud Detection — complete training pipeline.

1.  Threshold tuned on a HELD-OUT validation set — not the test set.
    This eliminates threshold leakage.
2.  scale_pos_weight set to the TRUE class ratio after SMOTE, not 1.
    This gives XGBoost honest signal about class importance.
3.  SMOTE ratio reduced from 0.10 → 0.30 so fewer synthetic samples
    are generated, keeping the distribution closer to reality.
4.  Precision floor enforced during threshold selection (min 3%).
    Prevents the model from flagging half the dataset for 30% recall.
5.  IsolationForest anomaly score added as an extra feature before
    training — this provides an unsupervised signal that helps the
    supervised models in near-random datasets.
6.  Three models: Logistic Regression, Random Forest, XGBoost.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import RandomForestClassifier, IsolationForest
from sklearn.metrics          import (
    average_precision_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    precision_recall_curve, fbeta_score, make_scorer,
)
from sklearn.linear_model     import LogisticRegression
import xgboost as xgb

# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
from src.transaction.preprocess import preprocess_pipeline

warnings.filterwarnings("ignore")

# CONSTANTS 

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.15      # fraction of train used for threshold tuning
SMOTE_RATIO   = 0.30      # fraud will be 30% of SMOTE'd train (was 10%)
FBETA         = 2.0       # recall weighted 2× precision
MIN_PRECISION = 0.03      # floor: never accept a threshold with <3% precision

TARGET_COLS = [
    "merchant_category",
    "sender_bank",
    "receiver_bank",
    "sender_state",
    "transaction_type",
    "category_combo",
    "bank_combo",
    "state_bank_combo",
    "device_network",
]
OHE_COLS = [
    "sender_age_group",
    "receiver_age_group",
    "day_of_week",
    "transaction_status",
    "device_type",
    "network_type",
]
SMOOTH_FACTOR = 50        # smaller → more trust in category means


#  HELPERS 

def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         beta: float = FBETA,
                         min_precision: float = MIN_PRECISION) -> float:
    """
    Sweep PR thresholds, enforce a precision floor, then maximise F-beta.
    Falls back to the unconstrained optimum if the floor eliminates everything.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Arrays are length n+1; thresholds is length n
    p = precisions[:-1]
    r = recalls[:-1]

    denom  = (beta ** 2 * p) + r + 1e-10
    f_beta = (1 + beta ** 2) * (p * r) / denom

    mask = p >= min_precision
    if mask.sum() > 0:
        best_idx = np.argmax(f_beta * mask)
    else:
        best_idx = np.argmax(f_beta)          # precision floor unachievable

    return float(thresholds[best_idx])


def _plot_confusion(cm: np.ndarray, name: str, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name.upper()}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{name}.png"), dpi=120)
    plt.close()


def _plot_pr_roc(y_test: np.ndarray, y_prob: np.ndarray,
                 name: str, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    axes[0].plot(rec, prec, lw=2, label=f"PR AUC = {pr_auc:.4f}")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall — {name.upper()}")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title(f"ROC Curve — {name.upper()}")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_roc_{name}.png"), dpi=120)
    plt.close()


#  MAIN CLASS 

class FraudDetector:

    def __init__(self, random_state: int = RANDOM_STATE,
                 model_dir: str = "models/transaction",
                 docs_dir:  str = "docs/transaction"):
        self.random_state   = random_state
        self.model_dir      = model_dir
        self.docs_dir       = docs_dir
        self.models:    dict = {}
        self.results:   dict = {}
        self.thresholds: dict = {}
        self.feature_names: list = []
        self.scaler: StandardScaler | None = None
        self.iso_forest: IsolationForest | None = None

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(docs_dir,  exist_ok=True)

    #  DATA PREPARATION 

    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        """
        1. Stratified 80/20 train/test split
        2. Leakage-free target encoding (train stats only)
        3. One-hot encoding + column alignment
        4. Drop residual string columns
        5. StandardScaler fit on train
        6. IsolationForest anomaly score as extra feature
        7. SMOTE on training set only
        Returns X_train_res, X_val_sc, X_test_sc, y_train_res, y_val, y_test
        """
        print("Train/test split ...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            stratify=y, random_state=self.random_state,
        )
        # Carve out a validation set from train for threshold tuning
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE,
            stratify=y_train, random_state=self.random_state,
        )

        # TARGET ENCODING (train fold only) 
        print("Leakage-free target encoding ...")
        global_mean = float(y_tr.mean())

        for col in TARGET_COLS:
            if col not in X_tr.columns:
                continue
            agg = y_tr.groupby(X_tr[col]).agg(["count", "mean"])
            smoothed = (
                (agg["count"] * agg["mean"] + SMOOTH_FACTOR * global_mean)
                / (agg["count"] + SMOOTH_FACTOR)
            )
            X_tr[f"{col}_risk"]  = X_tr[col].map(smoothed).fillna(global_mean)
            X_val[f"{col}_risk"] = X_val[col].map(smoothed).fillna(global_mean)
            X_test[f"{col}_risk"]= X_test[col].map(smoothed).fillna(global_mean)

        #  ONE-HOT ENCODING 
        print("One-hot encoding ...")
        existing_ohe = [c for c in OHE_COLS if c in X_tr.columns]
        X_tr   = pd.get_dummies(X_tr,   columns=existing_ohe, drop_first=True)
        X_val  = pd.get_dummies(X_val,  columns=existing_ohe, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=existing_ohe, drop_first=True)

        X_tr, X_val  = X_tr.align(X_val,  join="left", axis=1, fill_value=0)
        X_tr, X_test = X_tr.align(X_test, join="left", axis=1, fill_value=0)

        X_tr   = X_tr.select_dtypes(exclude=["object"])
        X_val  = X_val.select_dtypes(exclude=["object"])
        X_test = X_test.select_dtypes(exclude=["object"])

        self.feature_names = list(X_tr.columns)

        # SCALING 
        scaler = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_tr)
        X_val_sc  = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)
        self.scaler = scaler

        # ISOLATION FOREST (unsupervised anomaly score)
        print("Fitting IsolationForest anomaly detector ...")
        # Estimated fraud contamination from EDA
        true_fraud_rate = float(y_tr.mean())
        iso = IsolationForest(
            n_estimators=300,
            contamination=max(true_fraud_rate, 0.001),
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso.fit(X_tr_sc)
        self.iso_forest = iso

        # Anomaly score: higher = more anomalous (negated decision_function)
        iso_tr   = (-iso.decision_function(X_tr_sc)).reshape(-1, 1)
        iso_val  = (-iso.decision_function(X_val_sc)).reshape(-1, 1)
        iso_test = (-iso.decision_function(X_test_sc)).reshape(-1, 1)

        X_tr_sc   = np.hstack([X_tr_sc,   iso_tr])
        X_val_sc  = np.hstack([X_val_sc,  iso_val])
        X_test_sc = np.hstack([X_test_sc, iso_test])

        # SMOTE (train fold only) 
        print("Applying SMOTE ...")
        smote = SMOTE(
            sampling_strategy=SMOTE_RATIO,
            random_state=self.random_state,
            k_neighbors=5,
        )
        X_train_res, y_train_res = smote.fit_resample(X_tr_sc, y_tr)

        fraud_before = int(y_tr.sum())
        fraud_after  = int(y_train_res.sum())
        print(f"     Fraud: {fraud_before} → {fraud_after} "
              f"({fraud_after / len(y_train_res) * 100:.1f}% of train)")

        joblib.dump(scaler,              os.path.join(self.model_dir, "scaler.pkl"))
        joblib.dump(iso,                 os.path.join(self.model_dir, "iso_forest.pkl"))
        joblib.dump(self.feature_names,  os.path.join(self.model_dir, "feature_names.pkl"))

        return X_train_res, X_val_sc, X_test_sc, y_train_res, y_val, y_test

    # EVALUATION 

    def evaluate(self, name: str, model,
                 X_val:  np.ndarray, y_val,
                 X_test: np.ndarray, y_test) -> None:
        """
        1. Find optimal F-beta threshold on VALIDATION set (not test).
        2. Apply that threshold to TEST set for final numbers.
        """
        y_val_arr  = np.asarray(y_val)
        y_test_arr = np.asarray(y_test)

        # Threshold from validation
        val_prob  = model.predict_proba(X_val)[:, 1]
        best_thr  = _find_best_threshold(y_val_arr, val_prob)

        # Evaluate on test
        test_prob = model.predict_proba(X_test)[:, 1]
        y_pred    = (test_prob >= best_thr).astype(int)

        pr_auc   = average_precision_score(y_test_arr, test_prob)
        roc_auc  = roc_auc_score(y_test_arr, test_prob)
        report   = classification_report(y_test_arr, y_pred,
                                         output_dict=True, zero_division=0)
        cm       = confusion_matrix(y_test_arr, y_pred)
        f2       = fbeta_score(y_test_arr, y_pred, beta=FBETA, zero_division=0)

        fraud_key    = 1 if 1 in report else "1"
        prec_fraud   = report.get(fraud_key, {}).get("precision", 0.0)
        recall_fraud = report.get(fraud_key, {}).get("recall",    0.0)
        f1_fraud     = report.get(fraud_key, {}).get("f1-score",  0.0)

        self.results[name] = {
            "pr_auc":    pr_auc,
            "roc_auc":   roc_auc,
            "precision": prec_fraud,
            "recall":    recall_fraud,
            "f1":        f1_fraud,
            "f2":        f2,
            "confusion_matrix": cm,
        }
        self.thresholds[name] = best_thr

        tn, fp, fn, tp = cm.ravel()
        print(f"\n{'='*58}")
        print(f"  {name.upper()} | Threshold (from val): {best_thr:.4f}")
        print(f"{'='*58}")
        print(f"  PR-AUC : {pr_auc:.4f}   ROC-AUC : {roc_auc:.4f}")
        print(f"  Precision (Fraud) : {prec_fraud:.4f}")
        print(f"  Recall    (Fraud) : {recall_fraud:.4f}   << key metric")
        print(f"  F1 (Fraud)        : {f1_fraud:.4f}")
        print(f"  F2 (Fraud)        : {f2:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"{'='*58}\n")

        _plot_confusion(cm, name, self.docs_dir)
        _plot_pr_roc(y_test_arr, test_prob, name, self.docs_dir)
        joblib.dump(model, os.path.join(self.model_dir, f"{name}.pkl"))

    # INDIVIDUAL MODEL TRAINERS 

    def train_logistic(self, X_train, X_val, X_test,
                       y_train, y_val, y_test) -> None:
        print("Training Logistic Regression ...")
        # scale_pos_weight analogue: C controls regularisation strength
        # class_weight='balanced' provides per-sample weighting
        model = LogisticRegression(
            C=0.05,                 # stronger L2 → less overfit on SMOTE noise
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self.models["logistic"] = model
        self.evaluate("logistic", model, X_val, y_val, X_test, y_test)

    def train_random_forest(self, X_train, X_val, X_test,
                            y_train, y_val, y_test) -> None:
        print("Training Random Forest ...")
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,               # shallower → less overfit
            min_samples_leaf=20,        # each leaf needs 20 real examples
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self.models["rf"] = model
        self.evaluate("rf", model, X_val, y_val, X_test, y_test)

    def train_xgboost(self, X_train, X_val, X_test,
                      y_train, y_val, y_test) -> None:
        print("Training XGBoost ...")

        # Use val set for early stopping (not SMOTE'd)
        # scale_pos_weight: ratio of negatives to positives in train
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / max(pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=20,
            reg_alpha=2.0,
            reg_lambda=10.0,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            early_stopping_rounds=60,
            n_jobs=-1,
            random_state=self.random_state,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        best_iter = getattr(model, "best_iteration", model.n_estimators)
        print(f"  XGBoost best iteration: {best_iter}")
        self.models["xgboost"] = model
        self.evaluate("xgboost", model, X_val, y_val, X_test, y_test)

    #  COMPARISON PLOTS

    def plot_metrics_comparison(self) -> None:
        if not self.results:
            return

        metrics = ["recall", "precision", "f1", "f2", "pr_auc", "roc_auc"]
        df = pd.DataFrame(self.results).T[metrics]

        fig, ax = plt.subplots(figsize=(14, 6))
        x  = np.arange(len(df))
        w  = 0.13
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * w, df[metric], width=w,
                          label=metric.upper().replace("_", "-"), color=colors[i])
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + w * (len(metrics) - 1) / 2)
        ax.set_xticklabels([n.upper() for n in df.index], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison — UPI Fraud Detection",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.docs_dir, "model_comparison.png"), dpi=130)
        plt.close()

        # Recall spotlight
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(df)]
        bars = ax.bar(df.index.str.upper(), df["recall"], color=palette)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.4f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Recall (Fraud Class)")
        ax.set_title("Recall Comparison — Higher is Better for Fraud Detection",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.docs_dir, "recall_comparison.png"), dpi=130)
        plt.close()

        # PR-AUC spotlight
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(df.index.str.upper(), df["pr_auc"], color=palette)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002,
                    f"{h:.4f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("PR-AUC (Fraud Class)")
        ax.set_title("PR-AUC Comparison",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.docs_dir, "prauc_comparison.png"), dpi=130)
        plt.close()

        print("\n" + "=" * 65)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 65)
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print("=" * 65 + "\n")

    # ENTRY POINT 

    def fit_all(self, path: str) -> dict:
        print("\n" + "=" * 65)
        print("  UPI FRAUD DETECTION — TRAINING PIPELINE")
        print("=" * 65)

        print("\n[Step 1] Loading and preprocessing ...")
        X, y = preprocess_pipeline(path)
        print(f"  Dataset shape  : {X.shape}")
        print(f"  Fraud rate     : {y.mean():.4%} ({int(y.sum())} / {len(y)})")

        print("\n[Step 2] Preparing data ...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        print(f"  Train (after SMOTE) : {X_train.shape[0]:,} samples")
        print(f"  Val                 : {X_val.shape[0]:,} samples")
        print(f"  Test                : {X_test.shape[0]:,} samples")

        print("\n[Step 3] Training models ...")
        self.train_logistic(X_train, X_val, X_test, y_train, y_val, y_test)
        self.train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        self.train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

        print("\n[Step 4] Generating comparison plots ...")
        self.plot_metrics_comparison()

        print(f"\nModels → '{self.model_dir}/'")
        print(f"Plots  → '{self.docs_dir}/'\n")
        return self.results
