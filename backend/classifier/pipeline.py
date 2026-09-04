"""
classifier/pipeline.py
───────────────────────
Unified classification pipeline.

Decision logic:
  1. Try rule-based classifier first (fast, deterministic, no API cost)
  2. If rule returns None (ambiguous), call LLM classifier
  3. Combine: if rule gives low confidence and LLM gives higher, prefer LLM

Always returns a ClassificationResult.
"""

import logging
from typing import Optional

from classifier.rule_classifier import ClassificationResult, classify_by_rule
from classifier.llm_classifier import classify_by_llm

logger = logging.getLogger(__name__)

# Below this rule confidence threshold, we call the LLM to double-check
LLM_ASSIST_THRESHOLD = 0.80


def classify_failure(
    decline_code: str,
    failure_reason: Optional[str] = None,
    force_llm: bool = False,
) -> ClassificationResult:
    """
    Main entry point. Classifies a payment failure into a canonical category.

    Args:
        decline_code:   Machine-readable code from the payment processor.
        failure_reason: Optional free-text description (e.g. from processor message).
        force_llm:      If True, always use LLM (for testing/comparison).

    Returns:
        ClassificationResult with category, confidence, method, and reasoning.
    """
    if force_llm:
        result = classify_by_llm(decline_code, failure_reason)
        result.method = "llm"
        return result

    # Step 1: Rule-based
    rule_result = classify_by_rule(decline_code, failure_reason)

    if rule_result is not None and rule_result.confidence >= LLM_ASSIST_THRESHOLD:
        # High-confidence rule match — return immediately, no LLM cost
        logger.debug(
            "Rule classified '%s' → '%s' (conf=%.2f)",
            decline_code, rule_result.category, rule_result.confidence,
        )
        return rule_result

    # Step 2: LLM assist for ambiguous cases
    logger.debug(
        "Rule confidence %.2f < %.2f — invoking LLM for '%s'",
        rule_result.confidence if rule_result else 0.0,
        LLM_ASSIST_THRESHOLD,
        decline_code,
    )
    llm_result = classify_by_llm(decline_code, failure_reason)

    # Step 3: Pick the better result
    if rule_result is None:
        final = llm_result
        final.method = "llm"
    elif llm_result.confidence >= rule_result.confidence:
        final = llm_result
        final.method = "rule+llm"
        final.reasoning = (
            f"LLM overrode low-confidence rule result. "
            f"Rule: '{rule_result.category}' ({rule_result.confidence:.2f}). "
            f"LLM: {llm_result.reasoning}"
        )
    else:
        final = rule_result
        final.method = "rule+llm"
        final.reasoning = (
            f"Rule retained after LLM check. "
            f"Rule: '{rule_result.category}' ({rule_result.confidence:.2f}). "
            f"LLM agreed: '{llm_result.category}' ({llm_result.confidence:.2f})."
        )

    return final
