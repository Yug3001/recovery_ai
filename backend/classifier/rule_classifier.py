"""
classifier/rule_classifier.py
──────────────────────────────
Deterministic mapping from structured decline codes to failure categories.
When the decline code is well-known (unambiguous), this path runs with
confidence = 1.0 and no LLM call is needed.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    category: str           # canonical failure category
    confidence: float       # 0.0 – 1.0
    method: str             # 'rule' | 'llm' | 'rule+llm'
    reasoning: str          # human-readable explanation


# Exact decline-code → category mappings
_EXACT_MAP: dict[str, str] = {
    # Insufficient funds variants
    "insufficient_funds":        "insufficient_funds",
    "nsf":                       "insufficient_funds",
    "do_not_honor":              "insufficient_funds",   # often NSF in disguise
    "insufficient_balance":      "insufficient_funds",

    # Expired card
    "expired_card":              "expired_card",
    "card_expired":              "expired_card",
    "invalid_expiry":            "expired_card",

    # Temporary / processing errors
    "temp_error":                "temp_error",
    "temporary_error":           "temp_error",
    "gateway_timeout":           "temp_error",
    "network_error":             "temp_error",
    "processing_error":          "temp_error",
    "bank_unavailable":          "temp_error",
    "timeout":                   "temp_error",

    # Authentication required
    "auth_required":             "auth_required",
    "authentication_required":   "auth_required",
    "3ds_required":              "auth_required",
    "sca_required":              "auth_required",
    "strong_auth_required":      "auth_required",

    # Fraud / security flags
    "fraud_flag":                "fraud_flag",
    "security_violation":        "fraud_flag",
    "suspected_fraud":           "fraud_flag",
    "pickup_card":               "fraud_flag",       # card flagged by issuer
    "restricted_card":           "fraud_flag",

    # Hard declines
    "hard_decline":              "hard_decline",
    "do_not_retry":              "hard_decline",
    "card_blocked":              "hard_decline",
    "account_closed":            "hard_decline",
    "lost_card":                 "hard_decline",
    "stolen_card":               "hard_decline",
    "invalid_account":           "hard_decline",
    "invalid_card_number":       "hard_decline",

    # Unknown
    "unknown":                   "unknown",
    "generic_decline":           "unknown",
    "unspecified":               "unknown",
}

# Substring hints for fuzzy matching (lower-priority than exact)
_SUBSTRING_HINTS: list[tuple[str, str, float]] = [
    ("insufficient",    "insufficient_funds", 0.90),
    ("nsf",             "insufficient_funds", 0.90),
    ("balance",         "insufficient_funds", 0.80),
    ("expired",         "expired_card",       0.90),
    ("expir",           "expired_card",       0.85),
    ("timeout",         "temp_error",         0.90),
    ("network",         "temp_error",         0.85),
    ("temporary",       "temp_error",         0.90),
    ("processing",      "temp_error",         0.80),
    ("gateway",         "temp_error",         0.85),
    ("3d",              "auth_required",      0.85),
    ("sca",             "auth_required",      0.90),
    ("authenticat",     "auth_required",      0.90),
    ("fraud",           "fraud_flag",         0.95),
    ("suspicious",      "fraud_flag",         0.85),
    ("security",        "fraud_flag",         0.80),
    ("stolen",          "hard_decline",       0.95),
    ("lost",            "hard_decline",       0.90),
    ("blocked",         "hard_decline",       0.90),
    ("closed",          "hard_decline",       0.90),
    ("invalid",         "hard_decline",       0.80),
    ("do not honor",    "hard_decline",       0.85),
]

VALID_CATEGORIES = frozenset({
    "insufficient_funds",
    "expired_card",
    "temp_error",
    "auth_required",
    "fraud_flag",
    "hard_decline",
    "unknown",
})


def classify_by_rule(
    decline_code: str,
    failure_reason: Optional[str] = None,
) -> Optional[ClassificationResult]:
    """
    Try to classify deterministically.
    Returns None if the code is ambiguous enough to warrant LLM assistance.
    """
    # 1. Exact match on decline_code (highest confidence)
    normalized = decline_code.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _EXACT_MAP:
        category = _EXACT_MAP[normalized]
        return ClassificationResult(
            category=category,
            confidence=1.0,
            method="rule",
            reasoning=f"Exact match: decline_code='{decline_code}' → '{category}'",
        )

    # 2. Substring scan on decline_code + failure_reason combined text
    combined = f"{decline_code} {failure_reason or ''}".lower()
    best_category: Optional[str] = None
    best_confidence = 0.0
    for hint, cat, conf in _SUBSTRING_HINTS:
        if hint in combined and conf > best_confidence:
            best_category = cat
            best_confidence = conf

    if best_category and best_confidence >= 0.80:
        return ClassificationResult(
            category=best_category,
            confidence=best_confidence,
            method="rule",
            reasoning=(
                f"Substring match in combined text for hint '{hint}' "
                f"→ '{best_category}' (conf={best_confidence:.2f})"
            ),
        )

    # Ambiguous — signal caller to use LLM
    return None
