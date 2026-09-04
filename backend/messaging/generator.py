"""
messaging/generator.py
───────────────────────
Generates personalized customer-facing recovery messages (email + SMS)
using the Groq LLM, with template fallbacks when the API is unavailable.

One message pair (email + SMS) is generated per retry decision.
Messages are personalized to the failure type and customer context.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from groq import Groq, APIConnectionError, APIStatusError

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RecoveryMessage:
    email_subject: str
    email_body: str
    sms_body: str
    generated_by: str   # 'llm' | 'template'


# ── LLM prompt ────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a customer communication specialist for a subscription billing company.
Write empathetic, professional, and concise recovery messages for failed payments.
Keep the tone warm but not pushy. Never be accusatory.
Always include a clear call-to-action.
Respond ONLY with valid JSON — no extra text.
"""

_USER_TEMPLATE = """\
Generate a payment recovery message for this situation:
- Failure reason: {failure_category}
- Transaction amount: {currency}{amount:.2f}
- Retry will be attempted in: {retry_hours} hours
- Customer account age: {account_age_days} days

Return exactly this JSON structure:
{{
  "email_subject": "<subject line, max 60 chars>",
  "email_body": "<full email body, 80-150 words, include {{customer_name}} placeholder>",
  "sms_body": "<SMS version, max 160 chars, include clear action>"
}}
"""

# ── Fallback templates ─────────────────────────────────────────────────────────
_TEMPLATES: dict[str, RecoveryMessage] = {
    "insufficient_funds": RecoveryMessage(
        email_subject="Payment update needed for your subscription",
        email_body=(
            "Hi {customer_name},\n\n"
            "We were unable to process your recent payment of {currency}{amount:.2f} "
            "due to insufficient funds in your account.\n\n"
            "We'll automatically retry your payment in {retry_hours} hours. "
            "In the meantime, please ensure your account has sufficient funds or "
            "update your payment method to avoid any service interruption.\n\n"
            "If you have any questions, we're here to help.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Payment of {currency}{amount:.2f} failed (insufficient funds). "
            "We'll retry in {retry_hours}h. Please check your balance. "
            "Reply HELP for assistance."
        ),
        generated_by="template",
    ),
    "expired_card": RecoveryMessage(
        email_subject="Action required: Update your payment card",
        email_body=(
            "Hi {customer_name},\n\n"
            "Your payment of {currency}{amount:.2f} couldn't be processed because "
            "the card on file has expired.\n\n"
            "To keep your subscription active, please update your payment card "
            "by visiting your account settings. This only takes a minute.\n\n"
            "Once updated, we'll process your payment right away.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Your card has expired. Update it now to continue your subscription "
            "and process your {currency}{amount:.2f} payment. Visit account settings."
        ),
        generated_by="template",
    ),
    "temp_error": RecoveryMessage(
        email_subject="Brief payment hiccup — no action needed",
        email_body=(
            "Hi {customer_name},\n\n"
            "We experienced a temporary technical issue processing your payment "
            "of {currency}{amount:.2f}. This was on our end, not yours.\n\n"
            "We'll automatically retry your payment in {retry_hours} hours. "
            "No action is needed from you at this time.\n\n"
            "We apologize for any inconvenience.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Temporary payment issue ({currency}{amount:.2f}). We're retrying "
            "automatically in {retry_hours}h. No action needed. Sorry for the hiccup!"
        ),
        generated_by="template",
    ),
    "auth_required": RecoveryMessage(
        email_subject="Complete your payment verification",
        email_body=(
            "Hi {customer_name},\n\n"
            "Your bank requires additional verification to process your payment "
            "of {currency}{amount:.2f}.\n\n"
            "Please log in to your account and complete the verification step. "
            "This is a one-time security check required by your card issuer.\n\n"
            "We'll retry your payment in {retry_hours} hours once verification is complete.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Payment verification required for {currency}{amount:.2f}. "
            "Log in to complete the security check. We retry in {retry_hours}h."
        ),
        generated_by="template",
    ),
    "fraud_flag": RecoveryMessage(
        email_subject="Important: Your account requires attention",
        email_body=(
            "Hi {customer_name},\n\n"
            "We were unable to process your payment of {currency}{amount:.2f}. "
            "Your bank has flagged this transaction for security review.\n\n"
            "Please contact your bank or card issuer directly to resolve this, "
            "then contact our support team so we can assist you further.\n\n"
            "We take security seriously and are here to help.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Payment blocked by your bank for security review. "
            "Contact your bank then our support team. Amount: {currency}{amount:.2f}."
        ),
        generated_by="template",
    ),
    "hard_decline": RecoveryMessage(
        email_subject="Payment method issue — please update",
        email_body=(
            "Hi {customer_name},\n\n"
            "We were unable to process your payment of {currency}{amount:.2f} "
            "because your card was declined.\n\n"
            "Please update your payment method in your account settings or "
            "contact your bank for assistance.\n\n"
            "Our support team is available to help you resolve this.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Payment declined ({currency}{amount:.2f}). Please update your "
            "payment method or contact your bank."
        ),
        generated_by="template",
    ),
    "unknown": RecoveryMessage(
        email_subject="Payment issue with your subscription",
        email_body=(
            "Hi {customer_name},\n\n"
            "We encountered an issue processing your payment of {currency}{amount:.2f}.\n\n"
            "We'll attempt to retry in {retry_hours} hours. If this issue persists, "
            "please contact our support team and we'll help you resolve it quickly.\n\n"
            "Best regards,\nThe Billing Team"
        ),
        sms_body=(
            "Payment issue ({currency}{amount:.2f}). We'll retry in {retry_hours}h. "
            "Contact support if this continues."
        ),
        generated_by="template",
    ),
}


def _format_template(template: RecoveryMessage, **kwargs) -> RecoveryMessage:
    """Fill in template placeholders safely."""
    def _fmt(s: str) -> str:
        try:
            return s.format(**kwargs)
        except (KeyError, ValueError):
            return s

    return RecoveryMessage(
        email_subject=_fmt(template.email_subject),
        email_body=_fmt(template.email_body),
        sms_body=_fmt(template.sms_body),
        generated_by=template.generated_by,
    )


def generate_recovery_message(
    failure_category: str,
    amount: float,
    currency: str,
    retry_hours: Optional[int],
    account_age_days: int,
    customer_name: str = "Valued Customer",
) -> RecoveryMessage:
    """
    Generate a personalized recovery message for a failed payment.

    Tries LLM first; falls back to category-specific template on failure.

    Args:
        failure_category: Classified failure category.
        amount:           Transaction amount.
        currency:         Currency code (USD, EUR, etc.).
        retry_hours:      Planned retry timing in hours (None if no retry).
        account_age_days: Customer account age for personalization.
        customer_name:    Customer name for personalization.

    Returns:
        RecoveryMessage with email and SMS variants.
    """
    currency_symbol = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}.get(currency, currency + " ")
    retry_hours_str = str(retry_hours) if retry_hours else "shortly"

    # ── Try LLM first ─────────────────────────────────────────────────────────
    if settings.GROQ_API_KEY:
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            user_msg = _USER_TEMPLATE.format(
                failure_category=failure_category.replace("_", " "),
                currency=currency_symbol,
                amount=amount,
                retry_hours=retry_hours_str,
                account_age_days=account_age_days,
            )
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            import json
            parsed = json.loads(response.choices[0].message.content.strip())
            email_body = parsed.get("email_body", "").replace("{customer_name}", customer_name)
            return RecoveryMessage(
                email_subject=parsed.get("email_subject", "Payment update"),
                email_body=email_body,
                sms_body=parsed.get("sms_body", ""),
                generated_by="llm",
            )
        except (APIConnectionError, APIStatusError, Exception) as exc:
            logger.warning("LLM message generation failed (%s) — using template.", exc)

    # ── Template fallback ─────────────────────────────────────────────────────
    template = _TEMPLATES.get(failure_category, _TEMPLATES["unknown"])
    return _format_template(
        template,
        customer_name=customer_name,
        currency=currency_symbol,
        amount=amount,
        retry_hours=retry_hours_str,
        account_age_days=account_age_days,
    )
