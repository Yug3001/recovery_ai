"""
classifier/llm_classifier.py
─────────────────────────────
Groq LLM-assisted classification for ambiguous or free-text failure reasons.

The LLM is prompted to return structured JSON.  A strict schema validator
ensures we never accept hallucinated categories.  Falls back gracefully to
"unknown" if the Groq API is unavailable or returns invalid output.
"""

import json
import logging
from typing import Optional

from groq import Groq, APIConnectionError, APIStatusError

from classifier.rule_classifier import ClassificationResult, VALID_CATEGORIES
from config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a payment failure classifier for a subscription billing system.
Your job is to classify a payment failure into exactly one of these categories:

- insufficient_funds   : Customer account has insufficient balance
- expired_card         : The payment card has expired
- temp_error           : Temporary gateway, network, or processor error (likely retryable quickly)
- auth_required        : 3DS / SCA / additional authentication is needed
- fraud_flag           : Issuer or processor has flagged suspicious/fraudulent activity
- hard_decline         : Permanent decline — card blocked, closed, lost, stolen, or invalid account
- unknown              : None of the above can be confidently determined

Respond ONLY with a JSON object in this exact format — no extra text:
{
  "category": "<one of the 7 categories above>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explanation>"
}
"""

_USER_TEMPLATE = """\
Payment failure details:
- Decline code: {decline_code}
- Failure reason: {failure_reason}

Classify this failure.
"""


def classify_by_llm(
    decline_code: str,
    failure_reason: Optional[str] = None,
) -> ClassificationResult:
    """
    Use the Groq LLM to classify an ambiguous failure.
    Always returns a ClassificationResult (falls back to 'unknown' on error).
    """
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning 'unknown' for LLM classification.")
        return ClassificationResult(
            category="unknown",
            confidence=0.30,
            method="llm",
            reasoning="LLM unavailable (no API key); defaulted to unknown.",
        )

    client = Groq(api_key=settings.GROQ_API_KEY)
    user_msg = _USER_TEMPLATE.format(
        decline_code=decline_code or "not provided",
        failure_reason=failure_reason or "not provided",
    )

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,    # Low temp for deterministic classification
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        category = parsed.get("category", "unknown").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "LLM classification.")

        # Validate category
        if category not in VALID_CATEGORIES:
            logger.warning("LLM returned invalid category '%s'; defaulting to 'unknown'.", category)
            category = "unknown"
            confidence = 0.30
            reasoning = f"LLM returned invalid category; overridden to unknown. Original: {raw}"

        return ClassificationResult(
            category=category,
            confidence=min(max(confidence, 0.0), 1.0),
            method="llm",
            reasoning=reasoning,
        )

    except (APIConnectionError, APIStatusError) as exc:
        logger.error("Groq API error: %s — falling back to 'unknown'.", exc)
        return ClassificationResult(
            category="unknown",
            confidence=0.20,
            method="llm",
            reasoning=f"Groq API error: {exc}. Defaulted to unknown.",
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Failed to parse LLM response: %s", exc)
        return ClassificationResult(
            category="unknown",
            confidence=0.20,
            method="llm",
            reasoning=f"LLM response parse error: {exc}. Defaulted to unknown.",
        )
