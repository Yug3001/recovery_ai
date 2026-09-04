# Recovery AI — AI-Driven Payment Failure Recovery System

> **An end-to-end system that predicts the optimal retry timing and channel for failed recurring payments — and proves it recovers more revenue than a fixed-schedule baseline.**

---

## Evaluation Results (Real Numbers)

> These numbers are produced by `evaluation/evaluate.py` running on the held-out test split.
> Run the pipeline yourself and verify — see [Setup](#setup).

| Metric | AI Policy | Baseline |
|---|---|---|
| Recovery Rate | **~XX.X%** | ~XX.X% |
| Uplift | **+XX.X%** | — |
| Extra Revenue Recovered | **+$X,XXX** | — |

*Exact numbers generated at runtime — see `evaluation/results.json` and `evaluation/summary.txt` after running.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOVERY AI SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Synthetic Data Generator]  ──→  data/failed_payments.csv     │
│         data/generate_dataset.py                               │
│                  │                                              │
│                  ▼                                              │
│         [MySQL Database]                                        │
│          payment_failures                                       │
│          decisions                                              │
│          audit_logs                                             │
│          messages                                               │
│                  │                                              │
│        ┌─────────┴─────────┐                                   │
│        │                   │                                    │
│        ▼                   ▼                                    │
│ [Classification Pipeline]  [FastAPI Backend]                   │
│  ┌──────────────────┐      api/main.py                         │
│  │ Rule Classifier  │        POST /classify                    │
│  │ (deterministic)  │        POST /decide                      │
│  └────────┬─────────┘        GET  /decisions/{id}              │
│           │ ambiguous         GET  /audit-logs                  │
│           ▼                   GET  /evaluation                  │
│  ┌──────────────────┐                                          │
│  │  Groq LLM        │                                          │
│  │  (llama-3.3-70b) │                                          │
│  └────────┬─────────┘                                          │
│           │ ClassificationResult                                │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────┐              │
│  │          AI Retry-Recovery Model             │              │
│  │   GradientBoostingClassifier (sklearn)       │              │
│  │   Scores 7 timing candidates → best action   │              │
│  │   Safety gates: fraud/hard → escalate        │              │
│  └──────────────┬───────────────────────────────┘              │
│                 │                                               │
│        ┌────────┴────────┐                                     │
│        │                 │                                      │
│        ▼                 ▼                                      │
│ [AI Decision]   [Static Baseline]                              │
│  model/          baseline/rule_policy.py                       │
│  predictor.py    Fixed schedules per category                  │
│        │                                                        │
│        ▼                                                        │
│ [Messaging Generator]                                          │
│  Groq LLM → personalized email + SMS                          │
│  Template fallback if API unavailable                          │
│        │                                                        │
│        ▼                                                        │
│ [Audit Trail]                                                  │
│  Immutable append-only log in MySQL                            │
│        │                                                        │
│        ▼                                                        │
│ [Evaluation Engine]          [Streamlit Dashboard]             │
│  evaluation/evaluate.py  ──→  dashboard/app.py                │
│  AI vs Baseline on test set   📊 Evaluation                   │
│  results.json                 🔍 Decision Trail                │
│                               🚀 Live Demo                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

| # | Component | Location | Description |
|---|---|---|---|
| 1 | Synthetic Data | `data/generate_dataset.py` | 3,000 records with transparent label logic |
| 2 | Classification | `classifier/` | Rule-based + Groq LLM hybrid |
| 3 | AI Model | `model/` | GradientBoosting, trained on 80% split |
| 4 | Baseline | `baseline/rule_policy.py` | Fixed retry schedules |
| 5 | Messaging | `messaging/generator.py` | LLM-generated + template fallback |
| 6 | Audit Trail | `db/` + `api/main.py` | Immutable MySQL log |
| 7 | Evaluation | `evaluation/evaluate.py` | AI vs baseline on test split |
| 8 | Dashboard | `dashboard/app.py` | Streamlit 3-page UI |

---

## Setup

### Prerequisites
- Python 3.10+
- MySQL 8.0+ running locally
- Groq API key (get one free at [console.groq.com](https://console.groq.com))

### 1. Clone and install

```bash
git clone <your-repo-url>
cd "Recovery AI"

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
pip install -r frontend/requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env — set GROQ_API_KEY, MYSQL_PASSWORD, etc.
```

### 3. Create the MySQL database

```bash
python backend/create_db.py
```

### 4. Run the pipeline (in order)

```bash
# Step 1: Generate synthetic dataset (seeds DB + saves CSV)
python backend/data/generate_dataset.py

# Step 2: Train the ML model
python backend/model/train.py

# Step 3: Run evaluation (AI vs baseline)
python backend/evaluation/evaluate.py
```

### 5. Start the API server

```bash
# Terminal 1: Start the API
uvicorn backend.api.main:app --reload
```

### 6. Start the dashboard

```bash
# Terminal 2: Start the dashboard
streamlit run frontend/dashboard/app.py
```

### 7. Run tests

```bash
pytest backend/tests/ -v
```

---

## API Reference

Once the server is running, full docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health check |
| POST | `/classify` | Classify a failure event |
| POST | `/decide` | Full AI + baseline decision + message |
| GET | `/decisions/{txn_id}` | Audit trail for a transaction |
| GET | `/audit-logs` | Paginated audit log |
| GET | `/evaluation` | Evaluation metrics |

### Quick test

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-TEST-001",
    "customer_id": "CUST-0042",
    "amount": 99.99,
    "currency": "USD",
    "decline_code": "insufficient_funds",
    "past_success_count": 12,
    "past_failure_count": 1,
    "account_age_days": 450,
    "typical_payment_day": 15,
    "typical_payment_hour": 10,
    "avg_amount": 89.99
  }'
```

---

## File Structure

```
Recovery AI/
├── data/
│   ├── generate_dataset.py        # Synthetic data generator
│   ├── failed_payments.csv        # Generated dataset (gitignored)
│   └── README_data_generation.md  # Label logic documentation
├── db/
│   ├── schema.sql                 # MySQL schema
│   └── database.py                # SQLAlchemy ORM + helpers
├── classifier/
│   ├── rule_classifier.py         # Deterministic classification
│   ├── llm_classifier.py          # Groq LLM classification
│   └── pipeline.py                # Combined pipeline
├── model/
│   ├── features.py                # Feature engineering
│   ├── train.py                   # Model training
│   ├── predictor.py               # Inference + decision logic
│   └── retry_model.pkl            # Trained model (gitignored)
├── baseline/
│   └── rule_policy.py             # Static retry policy
├── messaging/
│   └── generator.py               # LLM + template messages
├── evaluation/
│   ├── evaluate.py                # AI vs baseline evaluation
│   ├── results.json               # Evaluation output (gitignored)
│   └── summary.txt                # Human-readable summary
├── api/
│   ├── main.py                    # FastAPI app
│   └── models.py                  # Pydantic models
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── tests/
│   ├── test_classifier.py
│   ├── test_baseline.py
│   └── test_model.py
├── config.py                      # Central settings
├── requirements.txt
├── .env.example
├── README.md
└── ARCHITECTURE.md
```

---

## Tech Stack

- **Backend**: Python 3.10, FastAPI, Uvicorn
- **ML**: scikit-learn (GradientBoostingClassifier), pandas, numpy, imbalanced-learn
- **LLM**: Groq API (llama-3.3-70b-versatile) — classification assist + message generation
- **Database**: MySQL 8 via SQLAlchemy + PyMySQL
- **Dashboard**: Streamlit + Plotly
- **Testing**: pytest

---

## Honest Disclaimers

- All evaluation numbers are based on **synthetic data with documented generation logic** — not real payment processor data
- Real-world recovery rates depend on bank policies, customer behavior, and processor specifics not modeled here
- The LLM is used for classification assist and message generation — the recovery decision is made by the sklearn model
- See `ARCHITECTURE.md` for full evidence status


streamlit run frontend/dashboard/app.py

cd "C:\Users\YUG\Desktop\Recovery AI"
python -m uvicorn backend.api.main:app --reload

cd "C:\Users\YUG\Desktop\Recovery AI"
python -m streamlit run frontend\dashboard\app.py

python -m pytest backend\tests\ -v