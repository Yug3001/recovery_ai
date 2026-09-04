"""
api/main.py
────────────
FastAPI application — all routes for the Recovery AI system.

Routes:
  POST /classify              — classify a failure event
  POST /decide                — full decision (classify + retry + message)
  GET  /decisions/{txn_id}    — get all decisions for a transaction
  GET  /audit-logs            — paginated audit log
  GET  /evaluation            — evaluation metrics (AI vs baseline)
  GET  /health                — health check
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    FailureEventRequest,
    ClassificationResponse,
    RetryDecisionResponse,
    MessageResponse,
    FullDecisionResponse,
    AuditLogEntry,
    EvaluationResponse,
    ScoredCandidate,
)
from classifier.pipeline import classify_failure
from baseline.rule_policy import apply_baseline_policy
from messaging.generator import generate_recovery_message
from db.database import (
    SessionLocal, PaymentFailure, Decision, AuditLog, Message,
    create_tables, append_audit_log, test_connection,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recovery AI — Payment Failure Recovery System",
    description=(
        "AI-driven payment failure recovery system that predicts the optimal "
        "retry timing and channel to maximize revenue recovery from failed "
        "recurring payments."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_JSON = Path(__file__).parent.parent / "evaluation" / "results.json"


# ── DB dependency ─────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Lazy model loader ─────────────────────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from model.predictor import RetryPredictor
            _predictor = RetryPredictor()
        except FileNotFoundError:
            logger.warning("Model not found — AI decisions will be unavailable.")
            _predictor = None
    return _predictor


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    if test_connection():
        create_tables()
        logger.info("Database ready.")
    else:
        logger.warning("Database not available — audit logging disabled.")
    get_predictor()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "db_connected": test_connection(),
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── POST /classify ────────────────────────────────────────────────────────────
@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
def classify_endpoint(req: FailureEventRequest, db: Session = Depends(get_db)):
    """
    Classify a failed payment into one of 7 canonical categories.
    Uses rule-based classifier first; falls back to Groq LLM for ambiguous inputs.
    """
    result = classify_failure(req.decline_code, req.failure_reason)

    # Audit log
    try:
        append_audit_log(db, req.transaction_id, "CLASSIFIED", {
            "category": result.category,
            "confidence": result.confidence,
            "method": result.method,
        })
        db.commit()
    except Exception as exc:
        logger.warning("Audit log failed: %s", exc)

    return ClassificationResponse(
        transaction_id=req.transaction_id,
        category=result.category,
        confidence=result.confidence,
        method=result.method,
        reasoning=result.reasoning,
    )


# ── POST /decide ──────────────────────────────────────────────────────────────
@app.post("/decide", response_model=FullDecisionResponse, tags=["Decisions"])
def decide_endpoint(req: FailureEventRequest, db: Session = Depends(get_db)):
    """
    Full pipeline: classify → AI retry decision → baseline decision → message.
    Returns both AI and baseline decisions side-by-side for comparison.
    Logs everything to the audit trail.
    """
    # 1. Classify
    classification = classify_failure(req.decline_code, req.failure_reason)

    # 2. AI decision
    predictor = get_predictor()
    if predictor:
        ai_raw = predictor.predict(
            failure_category=classification.category,
            decline_code=req.decline_code,
            past_success_count=req.past_success_count,
            past_failure_count=req.past_failure_count,
            account_age_days=req.account_age_days,
            typical_payment_day=req.typical_payment_day,
            typical_payment_hour=req.typical_payment_hour,
            avg_amount=req.avg_amount,
            amount=req.amount,
        )
        ai_decision = RetryDecisionResponse(
            transaction_id=req.transaction_id,
            policy="ai",
            failure_category=classification.category,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            action=ai_raw.action,
            retry_after_hours=ai_raw.retry_after_hours,
            retry_channel=ai_raw.retry_channel,
            recovery_probability=ai_raw.recovery_probability,
            model_version=ai_raw.model_version,
            reasoning=ai_raw.reasoning,
            scored_candidates=[ScoredCandidate(**c) for c in ai_raw.scored_candidates],
        )
    else:
        ai_decision = RetryDecisionResponse(
            transaction_id=req.transaction_id,
            policy="ai",
            failure_category=classification.category,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            action="escalate",
            retry_after_hours=None,
            retry_channel="manual_review",
            recovery_probability=None,
            model_version=None,
            reasoning="AI model not loaded — run `python model/train.py` first.",
        )

    # 3. Baseline decision
    bl_raw = apply_baseline_policy(classification.category)
    baseline_decision = RetryDecisionResponse(
        transaction_id=req.transaction_id,
        policy="baseline",
        failure_category=classification.category,
        classification_confidence=classification.confidence,
        classification_method=classification.method,
        action=bl_raw.action,
        retry_after_hours=bl_raw.retry_after_hours,
        retry_channel=bl_raw.retry_channel,
        recovery_probability=None,
        model_version=None,
        reasoning=bl_raw.reasoning,
    )

    # 4. Generate message
    msg = generate_recovery_message(
        failure_category=classification.category,
        amount=req.amount,
        currency=req.currency,
        retry_hours=ai_decision.retry_after_hours,
        account_age_days=req.account_age_days,
    )
    email_msg = MessageResponse(
        transaction_id=req.transaction_id,
        email_subject=msg.email_subject,
        email_body=msg.email_body,
        sms_body=msg.sms_body,
        generated_by=msg.generated_by,
    )

    # 5. Persist to DB
    try:
        # Store failure event if not already there
        existing = db.query(PaymentFailure).filter_by(transaction_id=req.transaction_id).first()
        if not existing:
            payment_failure = PaymentFailure(
                transaction_id=req.transaction_id,
                customer_id=req.customer_id,
                amount=req.amount,
                currency=req.currency,
                decline_code=req.decline_code,
                failure_reason=req.failure_reason,
                past_success_count=req.past_success_count,
                past_failure_count=req.past_failure_count,
                account_age_days=req.account_age_days,
                typical_payment_day=req.typical_payment_day,
                typical_payment_hour=req.typical_payment_hour,
                avg_amount=req.avg_amount,
                failure_timestamp=req.failure_timestamp or datetime.utcnow(),
                is_recoverable=0,  # unknown at this point
            )
            db.add(payment_failure)
            db.flush()

        # Store AI decision
        db.add(Decision(
            transaction_id=req.transaction_id,
            policy="ai",
            failure_category=classification.category,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            action=ai_decision.action,
            retry_after_hours=ai_decision.retry_after_hours,
            retry_channel=ai_decision.retry_channel,
            recovery_probability=ai_decision.recovery_probability,
            model_version=ai_decision.model_version,
            reasoning=ai_decision.reasoning,
        ))

        # Store baseline decision
        db.add(Decision(
            transaction_id=req.transaction_id,
            policy="baseline",
            failure_category=classification.category,
            classification_confidence=classification.confidence,
            classification_method=classification.method,
            action=bl_raw.action,
            retry_after_hours=bl_raw.retry_after_hours,
            retry_channel=bl_raw.retry_channel,
            reasoning=bl_raw.reasoning,
        ))

        # Store message
        db.add(Message(
            transaction_id=req.transaction_id,
            channel="email",
            subject=msg.email_subject,
            body=msg.email_body,
            generated_by=msg.generated_by,
        ))
        db.add(Message(
            transaction_id=req.transaction_id,
            channel="sms",
            body=msg.sms_body,
            generated_by=msg.generated_by,
        ))

        # Audit events
        append_audit_log(db, req.transaction_id, "CLASSIFIED", {
            "category": classification.category, "confidence": classification.confidence,
        })
        append_audit_log(db, req.transaction_id, "AI_DECISION_MADE", {
            "action": ai_decision.action, "retry_hours": ai_decision.retry_after_hours,
            "probability": ai_decision.recovery_probability,
        })
        append_audit_log(db, req.transaction_id, "BASELINE_DECISION_MADE", {
            "action": bl_raw.action, "retry_hours": bl_raw.retry_after_hours,
        })
        append_audit_log(db, req.transaction_id, "MESSAGE_GENERATED", {
            "generated_by": msg.generated_by, "channel": "email+sms",
        })
        db.commit()
    except Exception as exc:
        logger.warning("DB persist failed: %s", exc)
        db.rollback()

    return FullDecisionResponse(
        classification=ClassificationResponse(
            transaction_id=req.transaction_id,
            category=classification.category,
            confidence=classification.confidence,
            method=classification.method,
            reasoning=classification.reasoning,
        ),
        ai_decision=ai_decision,
        baseline_decision=baseline_decision,
        email_message=email_msg,
        sms_preview=msg.sms_body,
    )


# ── GET /decisions/{transaction_id} ───────────────────────────────────────────
@app.get("/decisions/{transaction_id}", tags=["Audit"])
def get_decisions(transaction_id: str, db: Session = Depends(get_db)):
    """Get all decisions and audit trail for a specific transaction."""
    decisions = db.query(Decision).filter_by(transaction_id=transaction_id).all()
    logs = db.query(AuditLog).filter_by(transaction_id=transaction_id).order_by(AuditLog.created_at).all()
    messages = db.query(Message).filter_by(transaction_id=transaction_id).all()

    if not decisions and not logs:
        raise HTTPException(status_code=404, detail=f"No records found for transaction {transaction_id}")

    return {
        "transaction_id": transaction_id,
        "decisions": [
            {
                "id": d.id,
                "policy": d.policy,
                "failure_category": d.failure_category,
                "classification_confidence": float(d.classification_confidence),
                "action": d.action,
                "retry_after_hours": d.retry_after_hours,
                "retry_channel": d.retry_channel,
                "recovery_probability": float(d.recovery_probability) if d.recovery_probability else None,
                "model_version": d.model_version,
                "reasoning": d.reasoning,
                "created_at": d.created_at.isoformat(),
            }
            for d in decisions
        ],
        "audit_trail": [
            {
                "id": log.id,
                "event_type": log.event_type,
                "actor": log.actor,
                "payload": log.payload,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "messages": [
            {
                "channel": m.channel,
                "subject": m.subject,
                "body": m.body,
                "generated_by": m.generated_by,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


# ── GET /audit-logs ───────────────────────────────────────────────────────────
@app.get("/audit-logs", tags=["Audit"])
def get_audit_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, le=200),
    event_type: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Paginated audit log. Filter by event_type if provided."""
    q = db.query(AuditLog)
    if event_type:
        q = q.filter(AuditLog.event_type == event_type)
    total = q.count()
    logs = q.order_by(AuditLog.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "logs": [
            {
                "id": log.id,
                "transaction_id": log.transaction_id,
                "event_type": log.event_type,
                "actor": log.actor,
                "payload": log.payload,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
    }


# ── GET /evaluation ───────────────────────────────────────────────────────────
@app.get("/evaluation", tags=["Evaluation"])
def get_evaluation():
    """
    Return the latest evaluation results (AI vs baseline comparison).
    Run `python evaluation/evaluate.py` to generate/refresh these results.
    """
    if not RESULTS_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run `python evaluation/evaluate.py` first.",
        )
    with open(RESULTS_JSON) as f:
        return json.load(f)
