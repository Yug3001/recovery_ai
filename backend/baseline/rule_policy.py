"""
baseline/rule_policy.py
────────────────────────
Static rule-based retry policy — the baseline for comparison.

This mirrors how most real subscription billing systems work today:
fixed retry schedules per decline category, with no per-customer
or per-timing ML-based optimization.

Policy v1.0:
┌─────────────────────┬──────────────────────────────┬─────────────────┐
│ Category            │ Retry schedule (hours)       │ Channel         │
├─────────────────────┼──────────────────────────────┼─────────────────┤
│ insufficient_funds  │ 48h → 120h → 168h            │ auto_charge     │
│ temp_error          │ 1h → 6h → 24h                │ auto_charge     │
│ auth_required       │ 24h → 72h                    │ email_prompt    │
│ expired_card        │ no retry                     │ email_prompt    │
│ fraud_flag          │ no retry                     │ manual_review   │
│ hard_decline        │ no retry                     │ manual_review   │
│ unknown             │ 24h → 72h                    │ sms_prompt      │
└─────────────────────┴──────────────────────────────┴─────────────────┘
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineDecision:
    action: str                       # 'retry' | 'escalate' | 'no_action'
    retry_after_hours: Optional[int]  # First retry window in hours
    retry_schedule: list[int]         # Full schedule (all planned retries)
    retry_channel: str
    policy_version: str = "v1.0"
    reasoning: str = ""


# ── Policy tables ─────────────────────────────────────────────────────────────
_RETRY_SCHEDULE: dict[str, list[int]] = {
    "insufficient_funds": [48, 120, 168],
    "temp_error":         [1, 6, 24],
    "auth_required":      [24, 72],
    "expired_card":       [],           # No retry — escalate to card update
    "fraud_flag":         [],           # No retry — manual review only
    "hard_decline":       [],           # No retry — permanent
    "unknown":            [24, 72],
}

_CHANNEL: dict[str, str] = {
    "insufficient_funds": "auto_charge",
    "temp_error":         "auto_charge",
    "auth_required":      "email_prompt",
    "expired_card":       "email_prompt",   # Still send a card-update email
    "fraud_flag":         "manual_review",
    "hard_decline":       "manual_review",
    "unknown":            "sms_prompt",
}

_REASONING: dict[str, str] = {
    "insufficient_funds": "Fixed schedule: retry at 48h, 120h, 168h (payday windows).",
    "temp_error":         "Fixed schedule: retry at 1h, 6h, 24h (transient errors clear quickly).",
    "auth_required":      "Fixed schedule: retry at 24h, 72h (customer completes authentication).",
    "expired_card":       "No retry: card is expired. Email sent to prompt card update.",
    "fraud_flag":         "No retry: security/fraud flag. Escalated to manual review.",
    "hard_decline":       "No retry: permanent hard decline. Escalated to manual review.",
    "unknown":            "Fixed schedule: retry at 24h, 72h (conservative default).",
}


def apply_baseline_policy(failure_category: str) -> BaselineDecision:
    """
    Apply the static rule-based retry policy to a classified failure.

    Args:
        failure_category: One of the 7 canonical failure categories.

    Returns:
        BaselineDecision with action, timing, and channel.
    """
    # Normalize
    cat = failure_category.strip().lower()
    if cat not in _RETRY_SCHEDULE:
        cat = "unknown"

    schedule = _RETRY_SCHEDULE[cat]
    channel = _CHANNEL[cat]
    reasoning = _REASONING[cat]

    if not schedule:
        # No retry categories
        action = "escalate" if cat in ("fraud_flag", "hard_decline") else "no_action"
        return BaselineDecision(
            action=action,
            retry_after_hours=None,
            retry_schedule=[],
            retry_channel=channel,
            reasoning=reasoning,
        )

    return BaselineDecision(
        action="retry",
        retry_after_hours=schedule[0],
        retry_schedule=schedule,
        retry_channel=channel,
        reasoning=reasoning,
    )


def baseline_recovers(
    failure_category: str,
    recovery_window_hours: Optional[float],
) -> bool:
    """
    Simulate whether the baseline policy would recover a given failure.

    The baseline succeeds if:
      - It schedules a retry, AND
      - At least one scheduled retry falls within the recovery_window_hours

    This is the key function used in evaluation to compute baseline recovery rate.

    Args:
        failure_category:    Classified failure category.
        recovery_window_hours: Ground-truth window from dataset (None if not recoverable).

    Returns:
        True if the baseline policy would recover this payment.
    """
    if recovery_window_hours is None:
        return False

    decision = apply_baseline_policy(failure_category)
    if decision.action != "retry":
        return False

    # Baseline recovers if it retries within the recovery window
    # We use the first retry in the schedule (baseline doesn't optimize)
    return any(h <= recovery_window_hours for h in decision.retry_schedule)
