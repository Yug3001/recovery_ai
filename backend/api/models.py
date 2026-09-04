"""
api/models.py
──────────────
Pydantic request/response models for the FastAPI backend.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class FailureEventRequest(BaseModel):
    """Payload to classify a payment failure and generate a recovery decision."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    decline_code: str = Field(..., description="Machine-readable decline code from processor")
    failure_reason: Optional[str] = Field(None, description="Free-text failure description")
    # Customer history
    past_success_count: int = Field(default=0, ge=0)
    past_failure_count: int = Field(default=0, ge=0)
    account_age_days: int = Field(default=0, ge=0)
    typical_payment_day: int = Field(default=15, ge=1, le=28)
    typical_payment_hour: int = Field(default=9, ge=0, le=23)
    avg_amount: float = Field(default=50.0, gt=0)
    failure_timestamp: Optional[datetime] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-DEMO-001",
                "customer_id": "CUST-0042",
                "amount": 99.99,
                "currency": "USD",
                "decline_code": "insufficient_funds",
                "failure_reason": "Card declined — insufficient funds",
                "past_success_count": 12,
                "past_failure_count": 1,
                "account_age_days": 450,
                "typical_payment_day": 15,
                "typical_payment_hour": 10,
                "avg_amount": 89.99,
            }
        }


# ── Responses ─────────────────────────────────────────────────────────────────

class ClassificationResponse(BaseModel):
    transaction_id: str
    category: str
    confidence: float
    method: str
    reasoning: str


class ScoredCandidate(BaseModel):
    hours: int
    probability: float


class RetryDecisionResponse(BaseModel):
    transaction_id: str
    policy: str
    failure_category: str
    classification_confidence: float
    classification_method: str
    action: str
    retry_after_hours: Optional[int]
    retry_channel: str
    recovery_probability: Optional[float]
    model_version: Optional[str]
    reasoning: str
    scored_candidates: list[ScoredCandidate] = []


class MessageResponse(BaseModel):
    transaction_id: str
    email_subject: str
    email_body: str
    sms_body: str
    generated_by: str


class FullDecisionResponse(BaseModel):
    classification: ClassificationResponse
    ai_decision: RetryDecisionResponse
    baseline_decision: RetryDecisionResponse
    email_message: MessageResponse
    sms_preview: str


class AuditLogEntry(BaseModel):
    id: int
    transaction_id: str
    event_type: str
    actor: str
    payload: Optional[dict]
    created_at: datetime


class EvaluationResponse(BaseModel):
    """Evaluation results — returned by /evaluation endpoint."""
    metadata: dict
    test_set: dict
    ai_policy: dict
    baseline_policy: dict
    comparison: dict
    category_breakdown: list[dict]
