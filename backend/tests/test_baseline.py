"""
tests/test_baseline.py
───────────────────────
Unit tests for the static rule-based baseline policy.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.rule_policy import apply_baseline_policy, baseline_recovers


class TestBaselinePolicy:
    def test_insufficient_funds_retries(self):
        d = apply_baseline_policy("insufficient_funds")
        assert d.action == "retry"
        assert d.retry_after_hours == 48
        assert 48 in d.retry_schedule
        assert 120 in d.retry_schedule
        assert d.retry_channel == "auto_charge"

    def test_temp_error_fast_retry(self):
        d = apply_baseline_policy("temp_error")
        assert d.action == "retry"
        assert d.retry_after_hours == 1
        assert d.retry_channel == "auto_charge"

    def test_auth_required_medium_retry(self):
        d = apply_baseline_policy("auth_required")
        assert d.action == "retry"
        assert d.retry_after_hours == 24

    def test_expired_card_no_retry(self):
        d = apply_baseline_policy("expired_card")
        assert d.action == "no_action"
        assert d.retry_after_hours is None
        assert d.retry_channel == "email_prompt"

    def test_fraud_flag_escalate(self):
        d = apply_baseline_policy("fraud_flag")
        assert d.action == "escalate"
        assert d.retry_after_hours is None
        assert d.retry_channel == "manual_review"

    def test_hard_decline_escalate(self):
        d = apply_baseline_policy("hard_decline")
        assert d.action == "escalate"
        assert d.retry_channel == "manual_review"

    def test_unknown_has_schedule(self):
        d = apply_baseline_policy("unknown")
        assert d.action == "retry"
        assert len(d.retry_schedule) > 0

    def test_unknown_category_defaults_to_unknown(self):
        d = apply_baseline_policy("some_random_code")
        assert d.action in ("retry", "escalate", "no_action")


class TestBaselineRecovery:
    def test_recovers_when_retry_within_window(self):
        # insufficient_funds with 48h first retry, window is 50h → should recover
        assert baseline_recovers("insufficient_funds", 50.0) is True

    def test_misses_when_retry_outside_window(self):
        # insufficient_funds first retry at 48h, but window is 30h → miss
        assert baseline_recovers("insufficient_funds", 30.0) is False

    def test_fraud_never_recovers(self):
        assert baseline_recovers("fraud_flag", 100.0) is False

    def test_hard_decline_never_recovers(self):
        assert baseline_recovers("hard_decline", 100.0) is False

    def test_none_window_not_recoverable(self):
        assert baseline_recovers("temp_error", None) is False

    def test_temp_error_recovers_within_1h(self):
        assert baseline_recovers("temp_error", 2.0) is True

    def test_temp_error_recovers_within_6h(self):
        assert baseline_recovers("temp_error", 10.0) is True
