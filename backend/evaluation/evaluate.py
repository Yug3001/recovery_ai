"""
evaluation/evaluate.py
───────────────────────
Runs both the AI policy and the static baseline on the held-out test split,
and produces a rigorous comparison with real numbers.

Usage:
    python evaluation/evaluate.py

Outputs:
    evaluation/results.json   — full metrics for dashboard
    evaluation/summary.txt    — human-readable summary printed to console

Methodology:
    - Uses ONLY the test split (dataset_split == 'test')
    - Ground-truth `is_recoverable` and `recovery_window_hours` are used
      ONLY for evaluation — they are never fed into the AI model during inference
    - AI policy: for each test record, score all timing candidates, pick best
    - Baseline: apply fixed schedule, check if any retry falls within the window
    - Recovery is counted as True if the policy's chosen timing ≤ recovery_window_hours
      (for recoverable records) or the policy correctly escalated (for non-recoverable)
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.rule_policy import apply_baseline_policy, baseline_recovers
from model.features import build_single_feature, RETRY_TIMING_CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_CSV = Path(__file__).parent.parent / "data" / "failed_payments.csv"
MODEL_PKL = Path(__file__).parent.parent / "model" / "retry_model.pkl"
RESULTS_OUT = Path(__file__).parent / "results.json"
SUMMARY_OUT = Path(__file__).parent / "summary.txt"


def load_model():
    import joblib
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model not found. Run `python model/train.py` first.")
    artifact = joblib.load(MODEL_PKL)
    return artifact["model"], artifact.get("version", "unknown")


def ai_policy_decision(
    model,
    row: pd.Series,
    min_prob_threshold: float = 0.30,
) -> dict:
    """
    Apply AI policy to a single test record.
    Returns dict: {action, retry_hours, recovery_probability}
    """
    from model.predictor import NO_RETRY_CATEGORIES

    category = row["decline_code"]
    if category in NO_RETRY_CATEGORIES:
        return {"action": "escalate", "retry_hours": None, "recovery_probability": 0.0}

    best_hours = None
    best_prob = 0.0

    for hours in RETRY_TIMING_CANDIDATES:
        X = build_single_feature(
            decline_code=row["decline_code"],
            past_success_count=int(row["past_success_count"]),
            past_failure_count=int(row["past_failure_count"]),
            account_age_days=int(row["account_age_days"]),
            typical_payment_day=int(row["typical_payment_day"]),
            typical_payment_hour=int(row["typical_payment_hour"]),
            avg_amount=float(row["avg_amount"]),
            amount=float(row["amount"]),
            retry_hours=float(hours),
        )
        prob = float(model.predict_proba(X)[0, 1])
        if prob > best_prob:
            best_prob = prob
            best_hours = hours

    if best_prob < min_prob_threshold:
        return {"action": "escalate", "retry_hours": None, "recovery_probability": best_prob}

    return {"action": "retry", "retry_hours": best_hours, "recovery_probability": best_prob}


def evaluate_record_ai(model, row: pd.Series) -> dict:
    """Returns whether AI policy recovers this record."""
    decision = ai_policy_decision(model, row)
    is_recoverable = bool(row["is_recoverable"])
    window = row["recovery_window_hours"]

    if not is_recoverable:
        # Correct outcome: escalate (don't waste retries)
        recovered = decision["action"] == "escalate"
    else:
        if decision["action"] == "retry" and decision["retry_hours"] is not None:
            recovered = decision["retry_hours"] <= window
        else:
            recovered = False

    return {
        "transaction_id": row["transaction_id"],
        "decline_code": row["decline_code"],
        "amount": float(row["amount"]),
        "currency": row["currency"],
        "is_recoverable": is_recoverable,
        "recovery_window_hours": window if not pd.isna(window) else None,
        "ai_action": decision["action"],
        "ai_retry_hours": decision["retry_hours"],
        "ai_recovery_prob": decision["recovery_probability"],
        "ai_recovered": recovered,
    }


def evaluate_record_baseline(row: pd.Series) -> dict:
    """Returns whether baseline policy recovers this record."""
    is_recoverable = bool(row["is_recoverable"])
    window = row["recovery_window_hours"]

    recovered = baseline_recovers(
        failure_category=row["decline_code"],
        recovery_window_hours=window if not pd.isna(window) else None,
    )
    decision = apply_baseline_policy(row["decline_code"])

    return {
        "transaction_id": row["transaction_id"],
        "decline_code": row["decline_code"],
        "amount": float(row["amount"]),
        "baseline_action": decision.action,
        "baseline_retry_hours": decision.retry_after_hours,
        "baseline_recovered": recovered,
    }


def compute_metrics(df_results: pd.DataFrame) -> dict:
    """Compute recovery rates, uplift, category breakdown, and revenue."""
    # ── Overall ───────────────────────────────────────────────────────────────
    total = len(df_results)
    recoverable_mask = df_results["is_recoverable"] == True

    # We measure recovery rate only on RECOVERABLE records
    # (the meaningful denominator — we can't recover what was never recoverable)
    recoverable = df_results[recoverable_mask]
    n_recoverable = len(recoverable)

    if n_recoverable == 0:
        raise ValueError("No recoverable records in test set — check dataset generation.")

    ai_rate = recoverable["ai_recovered"].mean()
    base_rate = recoverable["baseline_recovered"].mean()
    uplift_abs = ai_rate - base_rate
    uplift_pct = (uplift_abs / base_rate * 100) if base_rate > 0 else float("inf")

    # ── Revenue recovered ─────────────────────────────────────────────────────
    ai_revenue = recoverable.loc[recoverable["ai_recovered"], "amount"].sum()
    base_revenue = recoverable.loc[recoverable["baseline_recovered"], "amount"].sum()
    total_at_risk = recoverable["amount"].sum()
    revenue_uplift = ai_revenue - base_revenue

    # ── Category breakdown ────────────────────────────────────────────────────
    category_stats = []
    for cat, grp in recoverable.groupby("decline_code"):
        n = len(grp)
        ai_r = grp["ai_recovered"].mean()
        base_r = grp["baseline_recovered"].mean()
        up = (ai_r - base_r) * 100
        category_stats.append({
            "category": cat,
            "n_recoverable": n,
            "ai_recovery_rate": round(ai_r, 4),
            "baseline_recovery_rate": round(base_r, 4),
            "uplift_pct": round(up, 2),
            "ai_revenue_recovered": round(grp.loc[grp["ai_recovered"], "amount"].sum(), 2),
            "baseline_revenue_recovered": round(grp.loc[grp["baseline_recovered"], "amount"].sum(), 2),
        })

    # ── Full test set breakdown (including non-recoverable) ───────────────────
    return {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "test_set": {
            "total_records": total,
            "recoverable_records": n_recoverable,
            "non_recoverable_records": total - n_recoverable,
            "recovery_rate_in_population": round(n_recoverable / total, 4),
        },
        "ai_policy": {
            "recovery_rate": round(float(ai_rate), 4),
            "recovery_rate_pct": round(float(ai_rate) * 100, 2),
            "transactions_recovered": int(recoverable["ai_recovered"].sum()),
            "revenue_recovered": round(float(ai_revenue), 2),
        },
        "baseline_policy": {
            "recovery_rate": round(float(base_rate), 4),
            "recovery_rate_pct": round(float(base_rate) * 100, 2),
            "transactions_recovered": int(recoverable["baseline_recovered"].sum()),
            "revenue_recovered": round(float(base_revenue), 2),
        },
        "comparison": {
            "uplift_absolute": round(float(uplift_abs), 4),
            "uplift_pct": round(float(uplift_pct), 2),
            "extra_transactions_recovered": int(recoverable["ai_recovered"].sum() - recoverable["baseline_recovered"].sum()),
            "extra_revenue_recovered": round(float(revenue_uplift), 2),
            "total_revenue_at_risk": round(float(total_at_risk), 2),
        },
        "category_breakdown": sorted(category_stats, key=lambda x: -x["uplift_pct"]),
    }


def run_evaluation() -> dict:
    # ── Load data ─────────────────────────────────────────────────────────────
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_CSV}.")
    df = pd.read_csv(DATA_CSV)
    test_df = df[df["dataset_split"] == "test"].copy().reset_index(drop=True)
    logger.info("Evaluating on %d test records.", len(test_df))

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_version = load_model()
    logger.info("Model version: %s", model_version)

    # ── Evaluate per record ───────────────────────────────────────────────────
    ai_results = []
    base_results = []
    for _, row in test_df.iterrows():
        ai_results.append(evaluate_record_ai(model, row))
        base_results.append(evaluate_record_baseline(row))

    ai_df = pd.DataFrame(ai_results)
    base_df = pd.DataFrame(base_results)

    # Merge
    merged = ai_df.merge(base_df[["transaction_id", "baseline_action", "baseline_retry_hours", "baseline_recovered"]], on="transaction_id")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(merged)

    # ── Save detailed results ─────────────────────────────────────────────────
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "metadata": {
            "model_version": model_version,
            "n_test_records": len(test_df),
        },
        **metrics,
        "per_record_sample": merged.head(20).astype(object).where(pd.notna, None).to_dict("records"),
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(full_results, f, indent=2, default=str, allow_nan=False)
    logger.info("Results saved → %s", RESULTS_OUT)

    return full_results


def print_summary(results: dict) -> str:
    ai = results["ai_policy"]
    bl = results["baseline_policy"]
    cmp = results["comparison"]
    ts = results["test_set"]

    lines = [
        "=" * 65,
        "  AI-DRIVEN PAYMENT RECOVERY — EVALUATION RESULTS",
        "=" * 65,
        f"  Test records      : {ts['total_records']}",
        f"  Recoverable       : {ts['recoverable_records']} ({ts['recovery_rate_in_population']*100:.1f}% of test set)",
        "",
        "  ┌─────────────────────┬──────────────┬──────────────┐",
        "  │ Metric              │ AI Policy    │ Baseline     │",
        "  ├─────────────────────┼──────────────┼──────────────┤",
        f"  │ Recovery Rate       │ {ai['recovery_rate_pct']:>10.2f}% │ {bl['recovery_rate_pct']:>10.2f}% │",
        f"  │ Txns Recovered      │ {ai['transactions_recovered']:>12,} │ {bl['transactions_recovered']:>12,} │",
        f"  │ Revenue Recovered   │ ${ai['revenue_recovered']:>11,.2f} │ ${bl['revenue_recovered']:>11,.2f} │",
        "  └─────────────────────┴──────────────┴──────────────┘",
        "",
        f"  ► Uplift            : +{cmp['uplift_pct']:.2f}% ({cmp['uplift_absolute']*100:.2f} pp)",
        f"  ► Extra txns        : +{cmp['extra_transactions_recovered']}",
        f"  ► Extra revenue     : +${cmp['extra_revenue_recovered']:,.2f}",
        "",
        "  Category Breakdown (recoverable records only):",
        f"  {'Category':<22} {'AI Rate':>8} {'Base Rate':>10} {'Uplift':>8}",
        "  " + "-" * 52,
    ]
    for row in results["category_breakdown"]:
        lines.append(
            f"  {row['category']:<22} {row['ai_recovery_rate']*100:>7.1f}% "
            f"{row['baseline_recovery_rate']*100:>9.1f}% "
            f"{row['uplift_pct']:>+7.1f}%"
        )
    lines.append("=" * 65)
    summary = "\n".join(lines)
    print(summary)

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Summary saved → %s", SUMMARY_OUT)
    return summary


if __name__ == "__main__":
    results = run_evaluation()
    print_summary(results)
