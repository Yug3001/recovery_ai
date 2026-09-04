"""
model/train.py
───────────────
Train a GradientBoostingClassifier on the synthetic dataset to predict
recovery probability given failure category, customer history, and timing.

Usage:
    python model/train.py

Outputs:
    model/retry_model.pkl  — trained model + metadata (joblib)
    model/feature_importance.csv
    model/training_report.json
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import FEATURE_COLUMNS, build_features, RETRY_TIMING_CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUT = Path(__file__).parent / "retry_model.pkl"
IMPORTANCE_OUT = Path(__file__).parent / "feature_importance.csv"
REPORT_OUT = Path(__file__).parent / "training_report.json"
DATA_CSV = Path(__file__).parent.parent / "data" / "failed_payments.csv"

MODEL_VERSION = "v1.0"


def load_training_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV}. Run `python data/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_CSV)
    train_df = df[df["dataset_split"] == "train"].copy()
    logger.info("Training set: %d records", len(train_df))
    return train_df


def train() -> dict:
    df = load_training_data()

    # ── Target ────────────────────────────────────────────────────────────────
    y = df["is_recoverable"].astype(int)

    # ── Features ──────────────────────────────────────────────────────────────
    # For training, use the actual recovery_window_hours where recoverable,
    # and a neutral 24h where not (model learns timing matters positively).
    df["training_retry_hours"] = df["recovery_window_hours"].fillna(24)
    X = build_features(df, retry_hours_col="training_retry_hours")

    logger.info(
        "Class distribution: %d recoverable (%.1f%%), %d not",
        y.sum(), y.mean() * 100, (1 - y).sum(),
    )

    # ── Class imbalance handling ──────────────────────────────────────────────
    sample_weights = compute_sample_weight("balanced", y)

    # ── Model ─────────────────────────────────────────────────────────────────
    # GradientBoostingClassifier — interpretable via feature importances,
    # handles mixed feature types well, and is more robust than plain LR
    # on a synthetic dataset with non-linear interactions.
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4,
    )

    # ── Cross-validation ──────────────────────────────────────────────────────
    logger.info("Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", fit_params={"sample_weight": sample_weights})
    logger.info(
        "CV ROC-AUC: %.4f ± %.4f",
        cv_scores.mean(), cv_scores.std(),
    )

    # ── Final fit ─────────────────────────────────────────────────────────────
    logger.info("Fitting final model on full training set …")
    clf.fit(X, y, sample_weight=sample_weights)

    # ── Training metrics ──────────────────────────────────────────────────────
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    train_auc = roc_auc_score(y, y_prob)
    train_ap = average_precision_score(y, y_prob)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    logger.info("Train ROC-AUC: %.4f | AP: %.4f", train_auc, train_ap)
    logger.info("\n%s", classification_report(y, y_pred))

    # ── Feature importance ────────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_OUT, index=False)
    logger.info("Feature importance saved → %s", IMPORTANCE_OUT)

    # ── Persist model ─────────────────────────────────────────────────────────
    model_artifact = {
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat(),
        "train_auc": train_auc,
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
    }
    joblib.dump(model_artifact, MODEL_OUT)
    logger.info("Model saved → %s", MODEL_OUT)

    # ── Training report ───────────────────────────────────────────────────────
    training_report = {
        "version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat(),
        "n_training_samples": len(df),
        "class_balance": {
            "recoverable": int(y.sum()),
            "not_recoverable": int((1 - y).sum()),
            "recovery_rate_pct": round(y.mean() * 100, 2),
        },
        "cv_roc_auc": {
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
            "all_folds": [round(s, 4) for s in cv_scores.tolist()],
        },
        "train_metrics": {
            "roc_auc": round(train_auc, 4),
            "average_precision": round(train_ap, 4),
            "confusion_matrix": cm,
        },
        "classification_report": report,
        "top_10_features": importance_df.head(10)[["feature", "importance"]].to_dict("records"),
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(training_report, f, indent=2)
    logger.info("Training report saved → %s", REPORT_OUT)

    return training_report


if __name__ == "__main__":
    report = train()
    print("\n── Training Complete ─────────────────────────────────────────────")
    print(f"  CV ROC-AUC : {report['cv_roc_auc']['mean']:.4f} ± {report['cv_roc_auc']['std']:.4f}")
    print(f"  Train AUC  : {report['train_metrics']['roc_auc']:.4f}")
    print(f"  Model saved: model/retry_model.pkl")
    print(f"  Report     : model/training_report.json")
