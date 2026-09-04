# Recovery AI: Project Details and 5-Minute Video Script

## 1. Project Overview

**Recovery AI** is an end-to-end payment failure recovery system. It analyzes failed recurring payments, classifies the reason for failure, predicts the best retry time and channel, generates a customer message, stores an audit trail, and compares the AI policy with a fixed rule-based baseline.

The main business goal is to recover more legitimate failed payments while protecting customers and the business from unsafe retries. Fraud and permanent declines are never automatically retried.

This is a working prototype based on transparent synthetic data. It demonstrates the complete architecture and workflow without connecting to a real payment processor.

## 2. Main Technology Stack

- **Language:** Python 3.10+; currently tested with Python 3.12
- **API:** FastAPI and Uvicorn
- **Machine learning:** pandas, NumPy, scikit-learn `GradientBoostingClassifier`, joblib
- **LLM:** Groq API using `llama-3.3-70b-versatile`
- **Database:** MySQL 8, SQLAlchemy, and PyMySQL
- **Dashboard:** Streamlit, Plotly, and requests
- **Testing:** pytest

## 3. System Architecture

```text
Synthetic Data Generator
        |
        v
failed_payments.csv -----> Model Training -----> retry_model.pkl
        |                                             |
        |                                             v
        +-------------------------------> Evaluation Engine
                                                      |
                                                      v
                                                results.json

User/API request
        |
        v
Rule Classifier ---- high confidence ----> Classification
        |
        +---- ambiguous input ----> Groq LLM Classifier
                                      |
                                      v
                              AI Retry Predictor
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    AI Decision              Baseline Decision
                         |                         |
                         +------------+------------+
                                      v
                            Recovery Message Generator
                                      |
                         Groq message or template fallback
                                      |
                                      v
                              MySQL Audit Trail
                                      |
                                      v
                              Streamlit Dashboard
```

## 4. How the System Works

### Step 1: Generate payment failure data

`backend/data/generate_dataset.py` creates 3,000 synthetic failed-payment records. Each record contains:

- Transaction and customer identifiers
- Amount and currency
- Decline code and failure reason
- Previous successful and failed payment counts
- Account age
- Typical payment day and hour
- Average payment amount
- Recoverability label and recovery window
- Train/test dataset split

The generator uses a fixed random seed so the experiment can be reproduced. The data is split into 80% training records and 20% held-out test records.

### Step 2: Classify the failure

The classifier supports seven canonical categories:

1. `insufficient_funds`
2. `expired_card`
3. `temp_error`
4. `auth_required`
5. `fraud_flag`
6. `hard_decline`
7. `unknown`

The rule classifier runs first. It recognizes exact codes and useful text patterns with high confidence. Ambiguous input can be sent to the Groq LLM classifier. The LLM response is validated so only one of the seven approved categories is accepted. If Groq is unavailable, the system safely falls back to `unknown`.

### Step 3: Select an AI retry decision

`backend/model/train.py` trains a `GradientBoostingClassifier` using customer history, failure category, amount, and retry timing features.

The predictor tests seven possible retry windows:

```text
1 hour, 6 hours, 24 hours, 48 hours, 72 hours, 120 hours, 168 hours
```

It scores each candidate and chooses the timing with the highest recovery probability. If the best probability is below `0.30`, the system escalates instead of retrying.

Safety gates are applied before model scoring:

- `fraud_flag` always escalates to manual review.
- `hard_decline` always escalates to manual review.

The retry channel depends on the category. For example, insufficient funds and temporary errors use an automatic charge retry, while authentication and expired-card cases use a customer prompt.

### Step 4: Apply the baseline policy

The baseline is a fixed schedule representing a conventional billing system. It does not optimize timing for an individual customer.

Examples:

- Insufficient funds: first retry at 48 hours
- Temporary error: first retry at 1 hour
- Authentication required: first retry at 24 hours
- Expired card: no automatic retry; prompt the customer
- Fraud and hard decline: escalate

The API returns the AI and baseline decisions side by side so their behavior can be compared.

### Step 5: Generate a recovery message

The messaging module creates an email and SMS message using the failure category, amount, currency, retry timing, and account age. Groq can personalize the message. If the API is unavailable or no key is configured, built-in templates are used so the application still works end to end.

### Step 6: Store the audit trail

When MySQL is available, the system stores:

- Payment failure records
- AI and baseline decisions
- Email and SMS messages
- Immutable audit events such as `CLASSIFIED`, `AI_DECISION_MADE`, `BASELINE_DECISION_MADE`, and `MESSAGE_GENERATED`

The database uses foreign keys so decisions and messages belong to a payment failure record.

## 5. API Endpoints

The FastAPI service runs on port `8000`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Reports API, database, and model status |
| POST | `/classify` | Classifies a payment failure |
| POST | `/decide` | Runs the complete classification, AI, baseline, messaging, and persistence flow |
| GET | `/decisions/{transaction_id}` | Retrieves decisions, messages, and audit events |
| GET | `/audit-logs` | Returns paginated audit events |
| GET | `/evaluation` | Returns the latest AI-versus-baseline metrics |

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 6. Evaluation Results

The current evaluation uses 600 held-out test records, including 366 recoverable records.

| Metric | AI Policy | Baseline |
|---|---:|---:|
| Recovery rate | 89.62% | 87.43% |
| Transactions recovered | 328 | 320 |
| Revenue recovered | $32,558.70 | $30,253.80 |

The AI policy recovered:

- **8 additional transactions**
- **$2,304.90 additional simulated revenue**
- **2.19 percentage points of absolute uplift**
- **2.50% relative uplift**

Recovery rates are calculated on recoverable records only. The numbers are simulated and are not claims about real-world payment performance.

## 7. Dashboard Pages

The Streamlit dashboard runs on port `8501` and has three pages.

### Evaluation

Displays test-set size, recoverable population, AI and baseline recovery rates, revenue charts, category breakdown, and methodology notes.

### Decision Trail

Looks up a transaction ID and displays its AI decision, baseline decision, generated messages, and audit events from MySQL.

### Live Demo

Allows a user to submit a payment failure event through a form. It shows classification, confidence, classification method, AI timing scores, baseline timing, recovery channel, reasoning, and generated email/SMS output.

The sidebar also displays API, database, and model status.

## 8. Five-Minute Video Script

### 0:00-0:30 - Introduce the project

> Hello, this is Recovery AI, an AI-driven payment failure recovery system. The purpose is to decide whether a failed recurring payment should be retried, when it should be retried, and which communication channel should be used. The system also includes safety rules, database auditing, a machine-learning model, an evaluation engine, and a Streamlit dashboard.

**Screen:** Show the repository root and the project folders.

### 0:30-1:15 - Explain the architecture

> The project has a data-generation layer, a hybrid classification pipeline, a machine-learning retry predictor, a fixed baseline policy, a messaging generator, a MySQL audit trail, a FastAPI backend, and a Streamlit frontend. A payment failure enters through the API or dashboard. It is classified first, then scored by the AI predictor. The baseline is calculated at the same time. Finally, a customer message is generated and the complete result is saved to the database.

**Screen:** Open `ARCHITECTURE.md` and briefly show the component diagram.

### 1:15-2:00 - Explain the data and model

> The data generator creates 3,000 reproducible synthetic payment-failure records. The records include customer history, payment amount, decline category, and a recovery window. Eighty percent is used for training and twenty percent is held out for evaluation. The model is a Gradient Boosting Classifier. During inference, it tests seven retry windows from one hour to one week and selects the timing with the highest predicted probability. Fraud and hard declines bypass the model and always go to manual review.

**Screen:** Show `backend/data/generate_dataset.py`, `backend/model/train.py`, or the generated evaluation summary.

### 2:00-2:35 - Show the API health and evaluation

> The FastAPI service exposes health, classification, decision, audit, and evaluation endpoints. I will first check the health endpoint. It confirms that the API is running, the MySQL database is connected, and the trained model is loaded. The evaluation endpoint provides the same metrics used by the dashboard.

**Screen:** Open:

```text
http://127.0.0.1:8000/docs
```

Then show `/health` and `/evaluation`.

### 2:35-3:50 - Demonstrate the Live Demo

> Now I will use the Live Demo page. I will submit an insufficient-funds failure for a customer with a positive payment history. The system classifies the event, reports its confidence and method, and sends the customer data to the predictor. The AI scores all candidate timings and selects the best one. The baseline provides its fixed first retry timing. The dashboard also shows the recommended channel and the reasoning behind the decision.

> The system then generates an email and SMS message. If Groq is configured and available, the message is generated by the LLM. Otherwise, the built-in template fallback keeps the workflow operational.

**Screen:** Open the dashboard, choose **Live Demo**, submit the default form, and show:

- Classification category and confidence
- AI decision and recovery probability
- Scored timing candidates
- Baseline decision
- Email and SMS tabs

### 3:50-4:25 - Show safety and audit behavior

> A key safety feature is that fraud flags and hard declines are never automatically retried, regardless of the model probability. I can demonstrate this by selecting `fraud_flag`; the AI action becomes escalation with a manual-review channel. After a normal decision, the transaction and its two decisions, messages, and audit events are stored in MySQL.

**Screen:** Submit a fraud example, then open **Decision Trail** and search for the transaction ID.

### 4:25-5:00 - Present results and limitations

> On the held-out test set, the AI policy achieved an 89.62 percent recovery rate compared with 87.43 percent for the baseline. That represents eight additional recovered transactions and 2,304 dollars and 90 cents in simulated revenue. These results demonstrate the workflow and the advantage of personalized timing, but the data is synthetic. Real-world performance would need validation with payment processor data, customer behavior, bank policies, monitoring, and production security controls. This project is therefore a complete working prototype and evaluation framework, not a production payment-processing integration.

**Screen:** Return to the Evaluation page and show the KPI cards and comparison chart.

## 9. How to Run the Project Without a Virtual Environment

Run these commands from the project root:

```cmd
cd /d "C:\Users\YUG\Desktop\Recovery AI"
python -m pip install -r backend\requirements.txt
python -m pip install -r frontend\requirements.txt
```

Make sure MySQL is running and `.env` contains the correct database settings. Then prepare the data and model:

```cmd
python backend\create_db.py
python backend\data\generate_dataset.py
python backend\model\train.py
python backend\evaluation\evaluate.py
```

Start the API in Terminal 1:

```cmd
cd /d "C:\Users\YUG\Desktop\Recovery AI"
python -m uvicorn backend.api.main:app --reload
```

Start the dashboard in Terminal 2:

```cmd
cd /d "C:\Users\YUG\Desktop\Recovery AI"
python -m streamlit run frontend\dashboard\app.py
```

Open:

```text
Dashboard: http://localhost:8501
API docs:  http://localhost:8000/docs
```

Run tests with:

```cmd
python -m pytest backend\tests\ -v
```

For Groq message generation, use the compatible dependency versions from `backend\requirements.txt`. If the LLM is unavailable, the application uses safe classification and message fallbacks.

## 10. Important Reliability Fixes

The working version includes several fixes discovered during testing:

- Evaluation output converts missing values to JSON-safe `null` instead of returning invalid `NaN` values.
- Evaluation summaries are written using UTF-8 so Windows can save box-drawing and arrow characters.
- New payment-failure records are flushed before inserting dependent decisions and messages, satisfying MySQL foreign keys.
- SQLAlchemy builds the MySQL URL structurally, so passwords containing characters such as `@` are handled correctly.
- The dashboard shows API, database, and model availability in the sidebar.

## 11. Project File Guide

- `backend/data/generate_dataset.py`: creates the synthetic dataset
- `backend/data/README_data_generation.md`: documents label-generation logic
- `backend/classifier/rule_classifier.py`: deterministic classification
- `backend/classifier/llm_classifier.py`: Groq-assisted classification
- `backend/classifier/pipeline.py`: combines rule and LLM classification
- `backend/model/features.py`: feature engineering
- `backend/model/train.py`: trains and saves the model
- `backend/model/predictor.py`: selects retry timing and applies safety gates
- `backend/baseline/rule_policy.py`: fixed comparison policy
- `backend/messaging/generator.py`: LLM messages and template fallback
- `backend/db/database.py`: SQLAlchemy models and MySQL helpers
- `backend/api/main.py`: FastAPI application and endpoints
- `backend/api/models.py`: request and response schemas
- `backend/evaluation/evaluate.py`: held-out test evaluation
- `frontend/dashboard/app.py`: Streamlit dashboard
- `backend/tests/`: classifier, baseline, and model tests

## 12. Final One-Sentence Summary

Recovery AI is a transparent, safety-aware prototype that combines rules, an LLM, machine learning, MySQL auditing, FastAPI, and Streamlit to optimize recovery decisions for failed recurring payments and measure them against a conventional fixed retry policy.

## 13. How a Company Like Razorpay Could Use This Project

A payment platform such as Razorpay processes a large number of recurring and subscription payments for merchants. When a payment fails, the platform needs to decide whether to retry it, when to retry it, whether the customer needs to take an action, and how to communicate with the customer. Recovery AI demonstrates a decisioning layer that could sit after a payment failure event and before the next recovery action.

### Example production workflow

1. Razorpay's payment infrastructure emits a webhook or internal event for a failed recurring payment.
2. Recovery AI receives the event with the decline code, failure reason, amount, customer history, merchant context, and payment-method information.
3. The classifier maps processor-specific error codes and free-text reasons into standard categories.
4. The model estimates recovery probability for different retry windows.
5. Safety rules block retries for fraud indicators, permanently declined cards, and other restricted cases.
6. The system returns a recommended action: retry automatically, ask the customer to update or authenticate, or escalate to review.
7. The platform schedules the action and sends an appropriate email, SMS, or in-product notification.
8. The decision, reason, model version, and final outcome are stored for auditing and future model improvement.

### Problems this could solve

#### 1. Fixed retry schedules waste recovery opportunities

Many billing systems use the same retry schedule for every customer. A retry at the wrong time can fail even when the customer would have paid successfully later. Recovery AI evaluates multiple timing candidates using customer history, payment amount, failure type, and timing patterns.

#### 2. Repeated retries can frustrate customers

Retrying too frequently can create duplicate-looking attempts, customer complaints, and unnecessary bank declines. The probability threshold and category-specific timing help avoid low-value retry attempts.

#### 3. Different decline reasons need different actions

An insufficient-funds failure may be recoverable with a later automatic retry, while an expired card requires the customer to update the payment method. A fraud flag should be reviewed instead of retried. The classification and policy layers make these actions explicit.

#### 4. Fraud and hard declines require safety controls

An optimization model should never be allowed to override high-risk payment rules. Recovery AI uses unconditional safety gates for fraud flags and permanent declines, ensuring these cases are escalated rather than automatically charged again.

#### 5. Merchant revenue is lost when recoverable payments are abandoned

Failed subscription payments can cause involuntary churn. By identifying recoverable failures and selecting a more suitable retry time, a payment platform could help merchants recover revenue and retain subscribers.

#### 6. Support teams lack clear decision explanations

The system records classification reasoning, selected timing, recovery probability, policy version, and audit events. This gives merchant support and risk teams a traceable explanation for why an action was recommended.

#### 7. Customer communication is often generic

The messaging layer creates category-specific email and SMS messages. In a production platform, these messages could be adapted to the merchant's brand, language, customer preferences, and compliance rules.

#### 8. Payment teams need measurable proof of improvement

The evaluation engine compares the AI policy with a baseline on held-out data. A company could extend this into controlled A/B tests and measure recovery rate, recovered revenue, involuntary churn, customer complaints, retry cost, and false-positive risk.

### Razorpay-style use cases

- **Subscription billing:** recover failed recurring payments for SaaS, media, education, and membership merchants.
- **Invoice and mandate payments:** decide how to handle failed scheduled collections and customer re-authorization.
- **Merchant recovery APIs:** expose a recommended next action to merchants through an API or dashboard.
- **Smart dunning:** coordinate retries with email, SMS, WhatsApp, and in-product reminders.
- **Risk-aware recovery:** separate recoverable operational failures from fraud, disputes, and permanent declines.
- **Merchant analytics:** report recovery uplift and additional recovered revenue by merchant, payment method, geography, and decline category.

### What would be required before production deployment

This prototype would need additional production controls before a company such as Razorpay could use it with real payments:

- Integration with real payment processor webhooks and retry schedulers
- Real historical, consented, and privacy-compliant training data
- PCI-DSS, data-protection, access-control, and secrets-management reviews
- Idempotency, rate limits, retries, queues, and high-availability deployment
- Stronger fraud, dispute, and regulatory policy enforcement
- Model monitoring, drift detection, retraining, and rollback procedures
- Experiment controls to prove uplift without harming customers
- Human review workflows for uncertain and high-risk cases
- Message approval, localization, opt-out handling, and communication compliance
- Removal or masking of sensitive payment information from logs

The business value is not simply "use AI to retry more payments." The value is to make recovery decisions more selective, explainable, measurable, and customer-aware. Used with proper risk and compliance controls, this approach could help a payment platform increase merchant revenue recovery while reducing unnecessary retries and improving the customer experience.
