"""
model/features.py
──────────────────
Feature extraction and transformation for the retry-recovery ML model.

Features used:
  1. Failure category (one-hot encoded, 7 categories)
  2. Customer history:
     - past_success_count
     - past_failure_count
     - success_rate  (derived: past_success / total)
     - account_age_days
     - typical_payment_day
     - typical_payment_hour
     - avg_amount
  3. Proposed retry timing:
     - retry_hours_bucket (label: 0–5 corresponding to 1h,6h,24h,48h,120h,168h)
  4. Amount features:
     - amount (log-transformed)
     - amount_vs_avg_ratio  (amount / avg_amount)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORY_VALUES = [
    "insufficient_funds",
    "expired_card",
    "temp_error",
    "auth_required",
    "fraud_flag",
    "hard_decline",
    "unknown",
]

# Candidate retry timing buckets (hours) used during prediction
RETRY_TIMING_CANDIDATES = [1, 6, 24, 48, 72, 120, 168]

# The ordered list of feature column names produced by build_features()
FEATURE_COLUMNS = (
    # Category one-hot (7 dummies, drop_first=False for interpretability)
    [f"cat_{c}" for c in CATEGORY_VALUES]
    # Customer history
    + [
        "past_success_count",
        "past_failure_count",
        "success_rate",
        "account_age_days",
        "typical_payment_day",
        "typical_payment_hour",
        "avg_amount_log",
    ]
    # Transaction
    + ["amount_log", "amount_vs_avg_ratio"]
    # Retry timing
    + ["retry_hours_log"]
)


def build_features(
    df: pd.DataFrame,
    retry_hours_col: str = "recovery_window_hours",
) -> pd.DataFrame:
    """
    Transform a raw payments DataFrame into the ML feature matrix.

    Args:
        df: DataFrame with columns matching `payment_failures` schema.
        retry_hours_col: Column to use as the proposed retry timing feature.
                         Use 'recovery_window_hours' for training;
                         pass a constant column for inference.

    Returns:
        Feature DataFrame aligned to FEATURE_COLUMNS.
    """
    out = pd.DataFrame(index=df.index)

    # 1. One-hot encode failure category (using decline_code as proxy)
    for cat in CATEGORY_VALUES:
        out[f"cat_{cat}"] = (df["decline_code"] == cat).astype(int)

    # 2. Customer history features
    total = df["past_success_count"] + df["past_failure_count"] + 1
    out["past_success_count"] = df["past_success_count"].clip(0, 100)
    out["past_failure_count"] = df["past_failure_count"].clip(0, 50)
    out["success_rate"] = (df["past_success_count"] / total).clip(0, 1)
    out["account_age_days"] = df["account_age_days"].clip(0, 3000)
    out["typical_payment_day"] = df["typical_payment_day"].clip(1, 28)
    out["typical_payment_hour"] = df["typical_payment_hour"].clip(0, 23)
    out["avg_amount_log"] = np.log1p(df["avg_amount"].clip(0, 10000))

    # 3. Transaction amount
    out["amount_log"] = np.log1p(df["amount"].clip(0, 10000))
    avg_safe = df["avg_amount"].clip(lower=1.0)
    out["amount_vs_avg_ratio"] = (df["amount"] / avg_safe).clip(0, 10)

    # 4. Retry timing
    retry_hours = df[retry_hours_col].fillna(24).clip(0, 200)
    out["retry_hours_log"] = np.log1p(retry_hours)

    return out[FEATURE_COLUMNS]


def build_single_feature(
    decline_code: str,
    past_success_count: int,
    past_failure_count: int,
    account_age_days: int,
    typical_payment_day: int,
    typical_payment_hour: int,
    avg_amount: float,
    amount: float,
    retry_hours: float,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference.
    Returns a 1-row DataFrame aligned to FEATURE_COLUMNS.
    """
    row = {
        "decline_code": decline_code,
        "past_success_count": past_success_count,
        "past_failure_count": past_failure_count,
        "account_age_days": account_age_days,
        "typical_payment_day": typical_payment_day,
        "typical_payment_hour": typical_payment_hour,
        "avg_amount": avg_amount,
        "amount": amount,
        "recovery_window_hours": retry_hours,
    }
    df_row = pd.DataFrame([row])
    return build_features(df_row, retry_hours_col="recovery_window_hours")
