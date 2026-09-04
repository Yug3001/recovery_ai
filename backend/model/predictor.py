"""
model/predictor.py
───────────────────
Loads the trained model and selects the optimal retry action for a given
payment failure event.

Selection logic:
  1. If category is fraud_flag or hard_decline → escalate immediately (no model needed)
  2. Otherwise, score each candidate retry timing using the model
  3. Return the timing with the highest predicted recovery probability
  4. If max probability < MIN_PROBABILITY_THRESHOLD → escalate (not worth retrying)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from model.features import RETRY_TIMING_CANDIDATES, build_single_feature
from config import settings

logger = logging.getLogger(__name__)

# Below this predicted probability, we don't retry (escalate instead)
MIN_PROBABILITY_THRESHOLD = 0.30

# Categories that must NEVER be retried — escalate to manual review
NO_RETRY_CATEGORIES = frozenset({"fraud_flag", "hard_decline"})


@dataclass
class RetryDecision:
    action: str                     # 'retry' | 'escalate' | 'no_action'
    retry_after_hours: Optional[int]
    retry_channel: str              # 'auto_charge' | 'email_prompt' | 'sms_prompt' | 'manual_review'
    recovery_probability: float
    model_version: str
    reasoning: str
    scored_candidates: list[dict]   # all (hours, prob) pairs for transparency


# Preferred channel per failure category
_CHANNEL_MAP = {
    "insufficient_funds": "auto_charge",    # retry charge directly
    "temp_error":         "auto_charge",    # just retry
    "auth_required":      "email_prompt",   # customer needs to complete auth
    "expired_card":       "email_prompt",   # customer needs to update card
    "unknown":            "sms_prompt",
}


class RetryPredictor:
    def __init__(self, model_path: Optional[str] = None):
        path = Path(model_path or settings.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `python model/train.py` first."
            )
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.version = artifact.get("version", "unknown")
        self.feature_columns = artifact.get("feature_columns", [])
        logger.info("Loaded model version %s from %s", self.version, path)

    def predict(
        self,
        failure_category: str,
        decline_code: str,
        past_success_count: int,
        past_failure_count: int,
        account_age_days: int,
        typical_payment_day: int,
        typical_payment_hour: int,
        avg_amount: float,
        amount: float,
    ) -> RetryDecision:
        """
        Produce a retry decision for a single payment failure.

        Args:
            failure_category: Classified category (from classifier pipeline).
            decline_code:     Raw decline code (used in feature encoding).
            ... (customer history and transaction features)

        Returns:
            RetryDecision with optimal action, timing, channel, and reasoning.
        """
        # ── Safety gate: never retry fraud/hard declines ───────────────────────
        if failure_category in NO_RETRY_CATEGORIES:
            return RetryDecision(
                action="escalate",
                retry_after_hours=None,
                retry_channel="manual_review",
                recovery_probability=0.0,
                model_version=self.version,
                reasoning=(
                    f"Category '{failure_category}' is a no-retry safety gate. "
                    "Escalated to manual review without model scoring."
                ),
                scored_candidates=[],
            )

        # ── Score all candidate timings ────────────────────────────────────────
        scored = []
        for hours in RETRY_TIMING_CANDIDATES:
            X = build_single_feature(
                decline_code=decline_code,
                past_success_count=past_success_count,
                past_failure_count=past_failure_count,
                account_age_days=account_age_days,
                typical_payment_day=typical_payment_day,
                typical_payment_hour=typical_payment_hour,
                avg_amount=avg_amount,
                amount=amount,
                retry_hours=float(hours),
            )
            prob = float(self.model.predict_proba(X)[0, 1])
            scored.append({"hours": hours, "probability": round(prob, 4)})

        # ── Select best timing ─────────────────────────────────────────────────
        best = max(scored, key=lambda x: x["probability"])
        best_hours = best["hours"]
        best_prob = best["probability"]

        if best_prob < MIN_PROBABILITY_THRESHOLD:
            return RetryDecision(
                action="escalate",
                retry_after_hours=None,
                retry_channel="manual_review",
                recovery_probability=best_prob,
                model_version=self.version,
                reasoning=(
                    f"Best predicted recovery probability {best_prob:.2%} is below "
                    f"threshold {MIN_PROBABILITY_THRESHOLD:.0%}. Escalating."
                ),
                scored_candidates=scored,
            )

        channel = _CHANNEL_MAP.get(failure_category, "auto_charge")
        return RetryDecision(
            action="retry",
            retry_after_hours=best_hours,
            retry_channel=channel,
            recovery_probability=best_prob,
            model_version=self.version,
            reasoning=(
                f"Model selected {best_hours}h retry window with {best_prob:.2%} "
                f"predicted recovery probability (best of {len(scored)} candidates). "
                f"Channel: {channel}."
            ),
            scored_candidates=scored,
        )


# ── Module-level singleton (lazy-loaded) ──────────────────────────────────────
_predictor: Optional[RetryPredictor] = None


def get_predictor() -> RetryPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RetryPredictor()
    return _predictor
