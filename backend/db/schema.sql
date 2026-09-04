-- ============================================================
-- Recovery AI — MySQL Schema
-- ============================================================
-- Run once to initialize the database:
--   mysql -u root -p recovery_ai < db/schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS recovery_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE recovery_ai;

-- ── 1. payment_failures ───────────────────────────────────────────────────────
-- Raw failed-payment events ingested from the synthetic dataset or live feed.
CREATE TABLE IF NOT EXISTS payment_failures (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id          VARCHAR(64)     NOT NULL UNIQUE,
    customer_id             VARCHAR(64)     NOT NULL,
    amount                  DECIMAL(12, 2)  NOT NULL,
    currency                VARCHAR(8)      NOT NULL DEFAULT 'USD',
    decline_code            VARCHAR(64)     NOT NULL,
    failure_reason          TEXT,
    -- customer history features
    past_success_count      INT             NOT NULL DEFAULT 0,
    past_failure_count      INT             NOT NULL DEFAULT 0,
    account_age_days        INT             NOT NULL DEFAULT 0,
    typical_payment_day     TINYINT         NOT NULL DEFAULT 1,  -- 1-28
    typical_payment_hour    TINYINT         NOT NULL DEFAULT 9,  -- 0-23
    avg_amount              DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    -- ground truth (used for evaluation only, hidden from policy at inference time)
    is_recoverable          TINYINT(1)      NOT NULL DEFAULT 0,
    recovery_window_hours   INT,                                 -- NULL if not recoverable
    -- metadata
    failure_timestamp       DATETIME        NOT NULL,
    dataset_split           ENUM('train', 'test') NOT NULL DEFAULT 'train',
    created_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_customer_id   (customer_id),
    INDEX idx_decline_code  (decline_code),
    INDEX idx_split         (dataset_split),
    INDEX idx_failure_ts    (failure_timestamp)
) ENGINE=InnoDB;

-- ── 2. decisions ─────────────────────────────────────────────────────────────
-- One row per recovery decision made (classification + retry action).
CREATE TABLE IF NOT EXISTS decisions (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id          VARCHAR(64)     NOT NULL,
    policy                  ENUM('ai', 'baseline') NOT NULL,
    -- classification output
    failure_category        VARCHAR(64)     NOT NULL,
    classification_confidence DECIMAL(5, 4) NOT NULL,
    classification_method   VARCHAR(32)     NOT NULL,  -- 'rule' | 'llm' | 'rule+llm'
    -- retry action
    action                  ENUM('retry', 'escalate', 'no_action') NOT NULL,
    retry_after_hours       INT,                       -- NULL if not retry
    retry_channel           VARCHAR(32),               -- 'auto_charge' | 'email_prompt' | 'sms_prompt'
    -- model output
    recovery_probability    DECIMAL(5, 4),             -- NULL for baseline
    model_version           VARCHAR(32),
    policy_version          VARCHAR(32)     NOT NULL DEFAULT 'v1.0',
    -- evaluation outcome (filled in during evaluation run)
    predicted_recovered     TINYINT(1),
    actual_recovered        TINYINT(1),
    -- audit
    reasoning               TEXT,
    created_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_txn_id        (transaction_id),
    INDEX idx_policy        (policy),
    INDEX idx_category      (failure_category),
    INDEX idx_action        (action),
    INDEX idx_created_at    (created_at),
    FOREIGN KEY (transaction_id) REFERENCES payment_failures(transaction_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── 3. audit_logs ────────────────────────────────────────────────────────────
-- Immutable append-only event log.  Never UPDATE or DELETE rows here.
CREATE TABLE IF NOT EXISTS audit_logs (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id          VARCHAR(64)     NOT NULL,
    event_type              VARCHAR(64)     NOT NULL,  -- e.g. 'CLASSIFIED', 'RETRY_SCHEDULED', 'ESCALATED', 'MESSAGE_GENERATED'
    actor                   VARCHAR(64)     NOT NULL DEFAULT 'system',
    payload                 JSON,
    created_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_txn_id        (transaction_id),
    INDEX idx_event_type    (event_type),
    INDEX idx_created_at    (created_at)
) ENGINE=InnoDB;

-- ── 4. messages ───────────────────────────────────────────────────────────────
-- Generated customer-facing recovery messages.
CREATE TABLE IF NOT EXISTS messages (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id          VARCHAR(64)     NOT NULL,
    channel                 ENUM('email', 'sms') NOT NULL,
    subject                 VARCHAR(256),
    body                    TEXT            NOT NULL,
    generated_by            VARCHAR(32)     NOT NULL DEFAULT 'llm',  -- 'llm' | 'template'
    created_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_txn_id        (transaction_id),
    FOREIGN KEY (transaction_id) REFERENCES payment_failures(transaction_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;
