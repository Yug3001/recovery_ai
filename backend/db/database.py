"""
db/database.py
──────────────
SQLAlchemy engine + session factory wired to MySQL (PyMySQL driver).
All other modules import `get_db`, `engine`, and the ORM models from here.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    DECIMAL, JSON, Column, DateTime, Enum, ForeignKey,
    Integer, SmallInteger, String, Text, create_engine, text,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings

logger = logging.getLogger(__name__)

# ── Engine ─────────────────────────────────────────────────────────────────────
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=settings.MYSQL_USER,
    password=settings.MYSQL_PASSWORD,
    host=settings.MYSQL_HOST,
    port=settings.MYSQL_PORT,
    database=settings.MYSQL_DATABASE,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=(settings.LOG_LEVEL == "DEBUG"),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── ORM Base ───────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────────
class PaymentFailure(Base):
    __tablename__ = "payment_failures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), nullable=False, unique=True)
    customer_id = Column(String(64), nullable=False)
    amount = Column(DECIMAL(12, 2), nullable=False)
    currency = Column(String(8), nullable=False, default="USD")
    decline_code = Column(String(64), nullable=False)
    failure_reason = Column(Text)
    past_success_count = Column(Integer, nullable=False, default=0)
    past_failure_count = Column(Integer, nullable=False, default=0)
    account_age_days = Column(Integer, nullable=False, default=0)
    typical_payment_day = Column(SmallInteger, nullable=False, default=1)
    typical_payment_hour = Column(SmallInteger, nullable=False, default=9)
    avg_amount = Column(DECIMAL(12, 2), nullable=False, default=0.00)
    is_recoverable = Column(SmallInteger, nullable=False, default=0)
    recovery_window_hours = Column(Integer)
    failure_timestamp = Column(DateTime, nullable=False)
    dataset_split = Column(Enum("train", "test"), nullable=False, default="train")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("payment_failures.transaction_id", ondelete="CASCADE"), nullable=False)
    policy = Column(Enum("ai", "baseline"), nullable=False)
    failure_category = Column(String(64), nullable=False)
    classification_confidence = Column(DECIMAL(5, 4), nullable=False)
    classification_method = Column(String(32), nullable=False)
    action = Column(Enum("retry", "escalate", "no_action"), nullable=False)
    retry_after_hours = Column(Integer)
    retry_channel = Column(String(32))
    recovery_probability = Column(DECIMAL(5, 4))
    model_version = Column(String(32))
    policy_version = Column(String(32), nullable=False, default="v1.0")
    predicted_recovered = Column(SmallInteger)
    actual_recovered = Column(SmallInteger)
    reasoning = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), nullable=False)
    event_type = Column(String(64), nullable=False)
    actor = Column(String(64), nullable=False, default="system")
    payload = Column(JSON)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("payment_failures.transaction_id", ondelete="CASCADE"), nullable=False)
    channel = Column(Enum("email", "sms"), nullable=False)
    subject = Column(String(256))
    body = Column(Text, nullable=False)
    generated_by = Column(String(32), nullable=False, default="llm")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Helpers ────────────────────────────────────────────────────────────────────
def create_tables() -> None:
    """Create all tables if they don't exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context-manager session (use in FastAPI via Depends or manually)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def append_audit_log(
    db: Session,
    transaction_id: str,
    event_type: str,
    payload: Optional[dict] = None,
    actor: str = "system",
) -> AuditLog:
    """Append an immutable audit event. Never updates existing rows."""
    entry = AuditLog(
        transaction_id=transaction_id,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    db.add(entry)
    db.flush()
    return entry


def test_connection() -> bool:
    """Smoke-test the DB connection. Returns True if OK."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB connection failed: %s", exc)
        return False
