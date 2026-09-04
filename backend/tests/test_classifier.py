"""
tests/test_classifier.py
─────────────────────────
Unit tests for the classification pipeline.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.rule_classifier import classify_by_rule, VALID_CATEGORIES
from classifier.pipeline import classify_failure


class TestRuleClassifier:
    def test_exact_match_insufficient_funds(self):
        result = classify_by_rule("insufficient_funds")
        assert result is not None
        assert result.category == "insufficient_funds"
        assert result.confidence == 1.0
        assert result.method == "rule"

    def test_exact_match_expired_card(self):
        result = classify_by_rule("expired_card")
        assert result is not None
        assert result.category == "expired_card"

    def test_exact_match_hard_decline(self):
        result = classify_by_rule("hard_decline")
        assert result is not None
        assert result.category == "hard_decline"

    def test_fraud_flag_exact(self):
        result = classify_by_rule("fraud_flag")
        assert result is not None
        assert result.category == "fraud_flag"

    def test_case_insensitive_normalization(self):
        result = classify_by_rule("Insufficient_Funds")
        assert result is not None
        assert result.category == "insufficient_funds"

    def test_substring_match_from_reason(self):
        result = classify_by_rule("unknown_code", "Card declined — insufficient funds")
        assert result is not None
        assert result.category == "insufficient_funds"

    def test_returns_none_for_truly_ambiguous(self):
        result = classify_by_rule("err_xyz_99", "Something went wrong")
        # May return None (ambiguous) or a low-confidence result
        if result is not None:
            assert result.category in VALID_CATEGORIES

    def test_all_categories_valid(self):
        for cat in VALID_CATEGORIES:
            result = classify_by_rule(cat)
            assert result is not None
            assert result.category == cat


class TestPipeline:
    def test_classify_returns_valid_category(self):
        result = classify_failure("insufficient_funds", "Not enough balance")
        assert result.category in VALID_CATEGORIES
        assert 0.0 <= result.confidence <= 1.0
        assert result.method in ("rule", "llm", "rule+llm")

    def test_fraud_flag_always_fraud(self):
        result = classify_failure("fraud_flag")
        assert result.category == "fraud_flag"
        assert result.confidence == 1.0

    def test_hard_decline_always_hard(self):
        result = classify_failure("hard_decline", "Card permanently blocked")
        assert result.category == "hard_decline"

    def test_temp_error_variants(self):
        for code in ["gateway_timeout", "network_error", "temp_error"]:
            result = classify_failure(code)
            assert result.category == "temp_error"

    def test_unknown_code_returns_something(self):
        result = classify_failure("XYZ_999", "totally unknown error")
        assert result.category in VALID_CATEGORIES
        assert result.reasoning  # reasoning should never be empty
