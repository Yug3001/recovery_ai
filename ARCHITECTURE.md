# ARCHITECTURE.md — Recovery AI

## Component Breakdown

### 1. Synthetic Data Generator (`backend/data/generate_dataset.py`)

Generates 3,000 synthetic failed-payment records. The ground-truth `is_recoverable` label is constructed with explicit, documented probability rules (see `backend/data/README_data_generation.md`):
- `fraud_flag` and `hard_decline` → always non-recoverable (0%)
- `insufficient_funds` → 70% base rate, boosted by customer success history and account age
- `temp_error` → 85% base rate
- Recovery windows set per category with plausible timing (e.g., payday-aware 48–120h for insufficient funds)

**Why synthetic?** Real payment failure datasets require production PCI-DSS access. The synthetic data preserves the structure and challenge (class imbalance, category differences) of real data while allowing full transparency.

### 2. Classification Pipeline (`backend/classifier/`)

Two-stage hybrid:
1. **Rule-based** (`backend/classifier/rule_classifier.py`): Deterministic exact + substring matching. Returns confidence 1.0 for known codes, `None` for ambiguous inputs.
2. **LLM-assisted** (`backend/classifier/llm_classifier.py`): Groq `llama-3.3-70b-versatile` via `json_object` response format. Called only when rule confidence < 0.80 or rule returns None.

The pipeline picks the higher-confidence result. This minimizes API cost while maintaining coverage for free-text or non-standard decline descriptions.

### 3. AI Retry-Recovery Model (`backend/model/`)

**Model choice**: `GradientBoostingClassifier` (sklearn)
- Chosen over logistic regression because the relationship between timing, customer history, and recovery probability is non-linear (e.g., timing matters differently for insufficient_funds vs temp_error)
- Chosen over neural nets for interpretability (feature importances are directly readable)
- 200 estimators, learning_rate=0.05, max_depth=4, early stopping after 15 iterations without improvement
- Class imbalance handled with `compute_sample_weight("balanced", y)`
- 5-fold stratified cross-validation to detect overfitting

**Features**: Failure category (one-hot, 7 categories), customer history (success rate, account age, typical timing), transaction amount (log-transformed), and proposed retry timing (log-transformed hours).

**Inference**: For each failure event, score 7 candidate timing windows (1h, 6h, 24h, 48h, 72h, 120h, 168h) and select the highest-scoring one. Minimum probability threshold (0.30) below which we escalate instead.

**Safety constraints** (non-negotiable, implemented in `backend/model/predictor.py`):
- `fraud_flag` → always escalate, never scored
- `hard_decline` → always escalate, never scored

### 4. Static Baseline (`backend/baseline/rule_policy.py`)

Fixed retry schedules per category, mirroring how most billing systems work today:
- `insufficient_funds`: retry at 48h, 120h, 168h
- `temp_error`: retry at 1h, 6h, 24h
- `auth_required`: retry at 24h, 72h
- `expired_card`: no retry (email to prompt update)
- `fraud_flag`: escalate
- `hard_decline`: escalate

The baseline uses the first retry in the schedule (no timing optimization).

### 5. Messaging Generator (`backend/messaging/generator.py`)

Groq `llama-3.3-70b-versatile` generates personalized email + SMS recovery messages based on failure category, amount, and retry timing. Falls back to hardcoded templates (one per category) when the API is unavailable. Template fallbacks ensure the system works end-to-end without an API key.

### 6. Audit Trail (`backend/db/database.py`, `backend/api/main.py`)

Immutable append-only `audit_logs` table in MySQL. Every event (CLASSIFIED, AI_DECISION_MADE, BASELINE_DECISION_MADE, MESSAGE_GENERATED) is logged with full JSON payload. The table has no UPDATE or DELETE operations — every entry is permanent. Queryable via `GET /audit-logs` and `GET /decisions/{txn_id}`.

### 7. Evaluation Engine (`backend/evaluation/evaluate.py`)

Runs both policies on the **held-out test split only** (20% stratified, never touched during training). For each record:
- **AI**: Score all 7 timing candidates, pick best, check if chosen timing ≤ ground-truth `recovery_window_hours`
- **Baseline**: Apply fixed schedule, check if any scheduled retry falls within the window

Metrics computed:
- Recovery rate (on recoverable records only — correct denominator)
- Uplift absolute (percentage points) and relative (%)
- Per-category breakdown
- Simulated revenue recovered

### 8. Streamlit Dashboard (`frontend/dashboard/app.py`)

Three pages:
1. **Evaluation**: Charts and tables from `results.json` — the main deliverable
2. **Decision Trail**: Per-transaction audit lookup via API
3. **Live Demo**: Submit a failure event, see classification → AI/baseline decisions → message in real time

---

## Data Flow

```
Input failure event
      │
      ▼
[Rule Classifier] ──→ high confidence? ──→ [ClassificationResult]
      │                                            │
      │ low confidence                             │
      ▼                                            │
[Groq LLM]  ──────────────────────────────────────┘
                                                   │
                    ┌──────────────────────────────┘
                    │
                    ▼
          [Safety Gate Check]
               fraud? → ESCALATE
           hard_decline? → ESCALATE
                    │
                    ▼
          [Score 7 Timing Candidates]
            model.predict_proba(X)
                    │
                    ▼
          [Select Best Timing]
          prob >= 0.30? → RETRY
          prob <  0.30? → ESCALATE
                    │
                    ▼
          [Groq Message Generation]
          + Template fallback
                    │
                    ▼
          [MySQL Audit Log]
          (immutable append)
```

---

## Evidence Status

> This section is deliberately honest about what is and isn't validated.

| Claim | Status | Evidence |
|---|---|---|
| AI policy recovers more than baseline | ✅ Validated on synthetic test split | `backend/evaluation/results.json` |
| Classification pipeline handles 7 categories | ✅ Unit-tested | `backend/tests/test_classifier.py` |
| Baseline policy correctly implements fixed schedule | ✅ Unit-tested | `backend/tests/test_baseline.py` |
| Safety gates prevent fraud/hard-decline retries | ✅ Unit-tested | `backend/tests/test_model.py` |
| Model trained without test data leakage | ✅ Stratified 80/20 split, test only used in evaluation | `backend/data/generate_dataset.py`, `backend/evaluation/evaluate.py` |
| Audit log is immutable | ✅ No UPDATE/DELETE in codebase; verified by code review | `backend/db/database.py` |
| LLM classification accuracy on real data | ❌ Not validated | Tested only on synthetic decline codes/reasons |
| Real-world recovery rates match synthetic eval | ❌ Not validated | No real payment processor integration |
| Revenue numbers represent actual recovered revenue | ❌ Synthetic | Numbers are simulated from synthetic transaction amounts |

---

## Issues Faced and How I Solved Them

### 1. Class imbalance in training data
**Problem**: `fraud_flag` and `hard_decline` records are always `is_recoverable=0`, creating a hard class boundary the model could exploit. With class imbalance, the model initially just predicted "not recoverable" for everything.

**Solution**: Used `compute_sample_weight("balanced", y)` to upweight minority class (recoverable = 1). Also used 5-fold stratified CV to detect any fold where the model collapsed to majority-class prediction.

### 2. Model overfitting on synthetic data
**Problem**: The synthetic data has very clean signal (labels come directly from the generation function), so the model achieved near-perfect training accuracy. This raised concerns about overfitting to the synthetic distribution.

**Solution**: Constrained model complexity (`max_depth=4`, `min_samples_leaf=20`, `subsample=0.8`) and used early stopping (`n_iter_no_change=15`). The CV AUC gap vs training AUC is monitored in `training_report.json` to quantify overfitting.

### 3. LLM classification inconsistency
**Problem**: Early testing showed the Groq LLM would occasionally return category names with formatting variations (e.g., "Insufficient Funds" instead of "insufficient_funds") or invent new categories.

**Solution**: Added strict schema validation in `backend/classifier/llm_classifier.py` that checks the returned category against `VALID_CATEGORIES` frozenset. Invalid categories are overridden to "unknown" with logged warnings. Also used `temperature=0.1` and `response_format={"type": "json_object"}` for deterministic structured output.

### 4. Retry safety constraint implementation
**Problem**: An early version of the predictor would sometimes score `fraud_flag` events against the model, producing a low (but nonzero) probability and then escalating via the threshold — rather than having a hard no-retry guarantee.

**Solution**: Added an explicit `NO_RETRY_CATEGORIES` gate at the very start of `predictor.predict()` that returns an `escalate` decision before the model is ever called. This makes the safety constraint unconditional and code-reviewable, not threshold-dependent.

### 5. MySQL vs SQLite trade-off
**Problem**: SQLite is simpler but doesn't enforce foreign keys by default and has limited concurrent write support. MySQL requires more setup but is more representative of real fintech infrastructure.

**Solution**: Used MySQL via SQLAlchemy + PyMySQL with `pool_pre_ping=True` for connection health checks. Added `test_connection()` that gracefully degrades — the system produces the CSV and JSON outputs even if MySQL is unreachable, so the ML pipeline works without a database.

### 6. Evaluation denominator choice
**Problem**: If we compute recovery rate as (recovered / total records), non-recoverable records drag the rate down for both policies equally, making the comparison meaningless.

**Solution**: Recovery rate is computed on recoverable records only: `(recovered | is_recoverable==1) / (is_recoverable==1)`. This is the correct denominator — it measures "of the failures that could have been saved, how many did each policy actually save?"
