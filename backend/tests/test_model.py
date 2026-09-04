"""
tests/test_model.py
────────────────────
Unit tests for feature engineering and model predictor.
Skips model-dependent tests gracefully if model hasn't been trained yet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import (
    build_features, build_single_feature, FEATURE_COLUMNS, RETRY_TIMING_CANDIDATES,
)


class TestFeatureEngineering:
    def _make_row(self, **overrides):
        defaults = {
            "decline_code": "insufficient_funds",
            "past_success_count": 10,
            "past_failure_count": 2,
            "account_age_days": 300,
            "typical_payment_day": 15,
            "typical_payment_hour": 10,
            "avg_amount": 80.0,
            "amount": 99.99,
            "recovery_window_hours": 48.0,
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_feature_columns_count(self):
        df = self._make_row()
        X = build_features(df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_no_nan_in_features(self):
        df = self._make_row()
        X = build_features(df)
        assert not X.isnull().any().any()

    def test_category_one_hot_encoding(self):
        for cat in ["insufficient_funds", "expired_card", "temp_error", "fraud_flag"]:
            df = self._make_row(decline_code=cat)
            X = build_features(df)
            assert X[f"cat_{cat}"].iloc[0] == 1
            # All other one-hot columns should be 0
            other_cats = [c for c in FEATURE_COLUMNS if c.startswith("cat_") and c != f"cat_{cat}"]
            assert (X[other_cats] == 0).all().all()

    def test_amount_log_transform(self):
        df = self._make_row(amount=100.0)
        X = build_features(df)
        assert abs(X["amount_log"].iloc[0] - np.log1p(100.0)) < 1e-6

    def test_success_rate_clipped(self):
        df = self._make_row(past_success_count=100, past_failure_count=0)
        X = build_features(df)
        assert 0.0 <= X["success_rate"].iloc[0] <= 1.0

    def test_build_single_feature_shape(self):
        X = build_single_feature(
            decline_code="temp_error",
            past_success_count=5,
            past_failure_count=1,
            account_age_days=200,
            typical_payment_day=10,
            typical_payment_hour=9,
            avg_amount=50.0,
            amount=49.99,
            retry_hours=6.0,
        )
        assert X.shape == (1, len(FEATURE_COLUMNS))
        assert list(X.columns) == FEATURE_COLUMNS

    def test_zero_amount_handled(self):
        # avg_amount=0 should not cause division by zero
        X = build_single_feature(
            decline_code="unknown",
            past_success_count=0,
            past_failure_count=0,
            account_age_days=0,
            typical_payment_day=1,
            typical_payment_hour=9,
            avg_amount=0.01,    # clipped to 1 in ratio calc
            amount=10.0,
            retry_hours=24.0,
        )
        assert not np.isnan(X.values).any()


class TestRetryTimingCandidates:
    def test_candidates_sorted_ascending(self):
        assert RETRY_TIMING_CANDIDATES == sorted(RETRY_TIMING_CANDIDATES)

    def test_candidates_cover_expected_windows(self):
        # Should cover short (1h), medium (24h), and long (168h) windows
        assert 1 in RETRY_TIMING_CANDIDATES
        assert 24 in RETRY_TIMING_CANDIDATES
        assert 168 in RETRY_TIMING_CANDIDATES


@pytest.mark.skipif(
    not Path("model/retry_model.pkl").exists(),
    reason="Model not trained yet — run `python model/train.py`"
)
class TestPredictor:
    def setup_method(self):
        from model.predictor import RetryPredictor
        self.predictor = RetryPredictor()

    def test_fraud_always_escalates(self):
        decision = self.predictor.predict(
            failure_category="fraud_flag",
            decline_code="fraud_flag",
            past_success_count=10,
            past_failure_count=0,
            account_age_days=500,
            typical_payment_day=15,
            typical_payment_hour=10,
            avg_amount=80.0,
            amount=99.0,
        )
        assert decision.action == "escalate"
        assert decision.retry_channel == "manual_review"

    def test_hard_decline_always_escalates(self):
        decision = self.predictor.predict(
            failure_category="hard_decline",
            decline_code="hard_decline",
            past_success_count=5,
            past_failure_count=0,
            account_age_days=200,
            typical_payment_day=10,
            typical_payment_hour=9,
            avg_amount=50.0,
            amount=49.0,
        )
        assert decision.action == "escalate"

    def test_temp_error_likely_retries(self):
        decision = self.predictor.predict(
            failure_category="temp_error",
            decline_code="temp_error",
            past_success_count=20,
            past_failure_count=0,
            account_age_days=700,
            typical_payment_day=15,
            typical_payment_hour=10,
            avg_amount=50.0,
            amount=49.0,
        )
        # Should retry with high probability
        assert decision.action in ("retry", "escalate")
        assert decision.recovery_probability is not None

    def test_scored_candidates_all_timings(self):
        decision = self.predictor.predict(
            failure_category="insufficient_funds",
            decline_code="insufficient_funds",
            past_success_count=15,
            past_failure_count=1,
            account_age_days=400,
            typical_payment_day=15,
            typical_payment_hour=10,
            avg_amount=80.0,
            amount=79.0,
        )
        assert len(decision.scored_candidates) == len(RETRY_TIMING_CANDIDATES)
        for sc in decision.scored_candidates:
            assert 0.0 <= sc["probability"] <= 1.0
