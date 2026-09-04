# Data Generation Logic — Transparency Document

## Purpose
This document explains, line-by-line in prose, how the synthetic dataset in `failed_payments.csv` was constructed. A fintech evaluator should be able to reproduce the labels independently after reading this.

## File: `data/generate_dataset.py`

### Record Count
3,000 records, seeded with `RANDOM_SEED = 42` for full reproducibility.

### Customer Pool
500 unique customer IDs are pre-generated (`CUST-0000` … `CUST-0499`). Records randomly draw from this pool, so individual customers appear multiple times — mirroring real subscription data where the same customer can have multiple payment failures.

### Decline Code Distribution

| Code | Weight | Rationale |
|---|---|---|
| `insufficient_funds` | 30% | Most common real-world decline type |
| `expired_card` | 20% | Very common in subscription businesses |
| `temp_error` | 20% | Gateway/network noise |
| `auth_required` | 10% | 3DS and SCA-triggered declines |
| `fraud_flag` | 10% | Issuer-side fraud prevention |
| `hard_decline` | 5% | Permanently blocked cards |
| `unknown` | 5% | Generic/unclassified declines |

### Amount Distribution
Log-normal with `mean=4.2, sigma=0.8`, clipped to `[$5, $5000]`. This produces a right-skewed distribution resembling real subscription amounts (most between $20–$200, some high-value outliers).

### Failure Timestamp
Uniformly spread across calendar year 2024, with full hour/minute resolution.

---

## Ground-Truth Label Construction

### Step 1: Base Recovery Rate per Category

```
insufficient_funds  → 0.70
temp_error          → 0.85
auth_required       → 0.50
expired_card        → 0.20
fraud_flag          → 0.00   ← never recoverable
hard_decline        → 0.00   ← never recoverable
unknown             → 0.30
```

`fraud_flag` and `hard_decline` are deterministically **not recoverable** — no modifier can change this. This reflects real-world policy: retrying fraud-flagged or hard-declined transactions is both futile and a compliance risk.

### Step 2: Customer Context Modifiers

Applied to the base rate:

1. **Success rate bonus**: `(past_success_count / total_transactions) * 0.15`  
   A customer with 95% payment success history gets up to +0.15 added to their recovery probability.

2. **Account age bonus**: `+0.05` for accounts older than 365 days.  
   Older accounts are typically more reliable payers.

3. **High-value penalty**: `-0.10` for amounts above $500.  
   Higher-value failures are harder to recover (customers may dispute, banks may hold).

Final probability: `clamp(base + modifiers, 0.0, 1.0)`. A Bernoulli draw against this probability yields `is_recoverable`.

### Step 3: Recovery Window

Only populated when `is_recoverable == 1`:

| Category | Window | Logic |
|---|---|---|
| `insufficient_funds` | 48–120h | Payday-aware: `typical_payment_day` shifts the base window |
| `temp_error` | 1–24h | Network/gateway errors clear quickly |
| `auth_required` | 24–72h | Customer needs time to complete auth |
| `expired_card` | 72–168h | Customer needs time to update card |
| `unknown` | 24–72h | Conservative default |

### Step 4: Train/Test Split

Stratified 80/20 split using `decline_code` as the stratification key, ensuring all decline types are proportionally represented in both splits. The test split is held out and never used during model training.

---

## What This Dataset Does NOT Model

- Real payment processor APIs or actual transaction data
- Seasonal effects (e.g., December payment failures)
- Cross-customer contagion or network effects
- Actual bank-side decline reason codes (which vary by issuer)

These limitations are acknowledged in `ARCHITECTURE.md` under "Evidence Status."
