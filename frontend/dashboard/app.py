"""
dashboard/app.py
─────────────────
Streamlit evaluation dashboard for Recovery AI.

Three pages:
  1. 📊 Evaluation — AI vs baseline comparison with real numbers
  2. 🔍 Decision Trail — per-transaction audit view
  3. 🚀 Live Demo — submit a failure event and see results in real time
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
RESULTS_JSON = Path(__file__).parent.parent.parent / "backend" / "evaluation" / "results.json"

st.set_page_config(
    page_title="Recovery AI Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { color: #0f766e; }
    [data-testid="stMetricLabel"] { color: #475569; }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] { border-right: 1px solid #dbe4e8; }
    h1, h2, h3 { letter-spacing: 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/credit-card.png", width=60)
st.sidebar.title("Recovery AI")
st.sidebar.caption("AI-driven payment failure recovery system")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Evaluation", "🔍 Decision Trail", "🚀 Live Demo"],
)
st.sidebar.divider()
st.sidebar.caption("Built with FastAPI + scikit-learn + Groq LLM")
with st.sidebar:
    st.markdown("---")
    st.caption("System status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=2).json()
        st.success("API online")
        st.caption(
            f"Database: {'connected' if health.get('db_connected') else 'offline'}  "
            f"| Model: {'loaded' if health.get('model_loaded') else 'missing'}"
        )
    except requests.exceptions.RequestException:
        st.warning("API offline")


# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_results_from_file():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=30)
def load_results_from_api():
    try:
        r = requests.get(f"{API_BASE}/evaluation", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_results():
    # Try API first, fall back to file
    results = load_results_from_api()
    if not results:
        results = load_results_from_file()
    return results


def color_uplift(val):
    color = "green" if val > 0 else ("red" if val < 0 else "gray")
    return f"color: {color}; font-weight: bold"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Evaluation":
    st.title("📊 AI vs Baseline — Evaluation Results")
    st.caption(
        "Recovery rates computed on held-out test split (never seen during training). "
        "All numbers are from the evaluation engine — not hand-picked."
    )

    results = get_results()

    if not results:
        st.warning(
            "⚠️ No evaluation results found. Run:\n"
            "```bash\n"
            "python data/generate_dataset.py\n"
            "python model/train.py\n"
            "python evaluation/evaluate.py\n"
            "```"
        )
        st.stop()

    ai = results["ai_policy"]
    bl = results["baseline_policy"]
    cmp = results["comparison"]
    ts = results["test_set"]

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Records", f"{ts['total_records']:,}")
    c2.metric("Recoverable", f"{ts['recoverable_records']:,}", f"{ts['recovery_rate_in_population']*100:.1f}% of test set")
    c3.metric("AI Recovery Rate", f"{ai['recovery_rate_pct']:.2f}%", f"+{cmp['uplift_pct']:.2f}% vs baseline")
    c4.metric("Extra Txns Recovered", f"+{cmp['extra_transactions_recovered']:,}")
    c5.metric("Extra Revenue", f"${cmp['extra_revenue_recovered']:,.2f}", f"Total at risk: ${cmp['total_revenue_at_risk']:,.2f}")

    st.divider()

    # ── Side-by-side comparison ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recovery Rate Comparison")
        fig = go.Figure(data=[
            go.Bar(name="AI Policy", x=["AI Policy", "Baseline"], y=[ai["recovery_rate_pct"], bl["recovery_rate_pct"]],
                   marker_color=["#2196F3", "#9E9E9E"], text=[f"{ai['recovery_rate_pct']:.1f}%", f"{bl['recovery_rate_pct']:.1f}%"],
                   textposition="outside"),
        ])
        fig.update_layout(
            yaxis_title="Recovery Rate (%)", showlegend=False,
            yaxis=dict(range=[0, max(ai["recovery_rate_pct"], bl["recovery_rate_pct"]) * 1.25]),
            height=350, margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue Recovered")
        fig2 = go.Figure(data=[
            go.Bar(name="Revenue", x=["AI Policy", "Baseline"],
                   y=[ai["revenue_recovered"], bl["revenue_recovered"]],
                   marker_color=["#4CAF50", "#9E9E9E"],
                   text=[f"${ai['revenue_recovered']:,.0f}", f"${bl['revenue_recovered']:,.0f}"],
                   textposition="outside"),
        ])
        fig2.update_layout(
            yaxis_title="Revenue Recovered ($)", showlegend=False,
            height=350, margin=dict(t=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Category breakdown ────────────────────────────────────────────────────
    st.subheader("Category Breakdown (Recoverable Records Only)")
    cat_df = pd.DataFrame(results["category_breakdown"])
    cat_df["ai_recovery_rate_pct"] = (cat_df["ai_recovery_rate"] * 100).round(1)
    cat_df["baseline_recovery_rate_pct"] = (cat_df["baseline_recovery_rate"] * 100).round(1)

    # Grouped bar chart
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="AI Policy", x=cat_df["category"], y=cat_df["ai_recovery_rate_pct"],
        marker_color="#2196F3",
        text=cat_df["ai_recovery_rate_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        name="Baseline", x=cat_df["category"], y=cat_df["baseline_recovery_rate_pct"],
        marker_color="#9E9E9E",
        text=cat_df["baseline_recovery_rate_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig3.update_layout(
        barmode="group", yaxis_title="Recovery Rate (%)",
        xaxis_title="Failure Category", height=400, margin=dict(t=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Table
    display_df = cat_df[[
        "category", "n_recoverable",
        "ai_recovery_rate_pct", "baseline_recovery_rate_pct", "uplift_pct",
        "ai_revenue_recovered", "baseline_revenue_recovered",
    ]].rename(columns={
        "category": "Category",
        "n_recoverable": "# Recoverable",
        "ai_recovery_rate_pct": "AI Rate (%)",
        "baseline_recovery_rate_pct": "Baseline Rate (%)",
        "uplift_pct": "Uplift (%)",
        "ai_revenue_recovered": "AI Revenue ($)",
        "baseline_revenue_recovered": "Baseline Revenue ($)",
    })

    styled = display_df.style.applymap(color_uplift, subset=["Uplift (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📋 Evidence & Methodology Notes")
    st.info(
        "**What these numbers represent**: Recovery rates computed on a 20% held-out test split (600 records) "
        "using synthetic data generated with transparent, documented logic. "
        "The AI model was trained only on the training split (2,400 records). "
        "Ground-truth labels were constructed with explicit probability rules — see `data/README_data_generation.md`. "
        "\n\n"
        "**What this does NOT claim**: These numbers are based on synthetic evaluation. "
        "Real-world recovery rates depend on actual payment processor behavior, customer response rates, "
        "and bank policies — which this prototype does not model."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DECISION TRAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Decision Trail":
    st.title("🔍 Transaction Decision Trail")
    st.caption("Full audit trail for any transaction — every decision logged immutably.")

    txn_id = st.text_input("Enter Transaction ID", placeholder="TXN-DEMO-001")

    if txn_id:
        try:
            r = requests.get(f"{API_BASE}/decisions/{txn_id}", timeout=5)
            if r.status_code == 404:
                st.warning(f"No records found for transaction `{txn_id}`")
            elif r.status_code == 200:
                data = r.json()

                st.subheader(f"Transaction: `{txn_id}`")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Decisions**")
                    for d in data.get("decisions", []):
                        badge = "🤖 AI" if d["policy"] == "ai" else "📏 Baseline"
                        action_color = {"retry": "🟢", "escalate": "🔴", "no_action": "🟡"}.get(d["action"], "⚪")
                        with st.expander(f"{badge} — {action_color} {d['action'].upper()}", expanded=True):
                            st.json(d)

                with col2:
                    st.markdown("**Audit Trail**")
                    audit_df = pd.DataFrame(data.get("audit_trail", []))
                    if not audit_df.empty:
                        st.dataframe(audit_df[["event_type", "actor", "created_at", "payload"]], use_container_width=True)

                if data.get("messages"):
                    st.markdown("**Generated Messages**")
                    for msg in data["messages"]:
                        with st.expander(f"{'📧 Email' if msg['channel'] == 'email' else '📱 SMS'} — {msg.get('subject', '')}"):
                            st.text(msg["body"])
            else:
                st.error(f"API error: {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI server not running. Start it with:\n```bash\nuvicorn api.main:app --reload\n```")

    else:
        # Show recent audit logs
        st.subheader("Recent Audit Events")
        try:
            r = requests.get(f"{API_BASE}/audit-logs?page_size=20", timeout=5)
            if r.status_code == 200:
                logs = r.json().get("logs", [])
                if logs:
                    log_df = pd.DataFrame(logs)[["transaction_id", "event_type", "actor", "created_at"]]
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No audit events yet. Submit a decision via the Live Demo page or the API.")
        except requests.exceptions.ConnectionError:
            st.info("Start the FastAPI server to see live audit events.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Live Demo":
    st.title("🚀 Live Decision Demo")
    st.caption("Submit a payment failure event and see the AI pipeline run in real time.")

    DECLINE_OPTIONS = [
        "insufficient_funds", "expired_card", "temp_error",
        "auth_required", "fraud_flag", "hard_decline", "unknown",
    ]

    with st.form("demo_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            txn_id = st.text_input("Transaction ID", value="TXN-DEMO-001")
            customer_id = st.text_input("Customer ID", value="CUST-0001")
            amount = st.number_input("Amount ($)", min_value=1.0, value=99.99, step=0.01)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])
        with col2:
            decline_code = st.selectbox("Decline Code", DECLINE_OPTIONS)
            failure_reason = st.text_area("Failure Reason (optional)", value="", height=80)
            past_success = st.number_input("Past Successes", min_value=0, value=12)
            past_failures = st.number_input("Past Failures", min_value=0, value=1)
        with col3:
            account_age = st.number_input("Account Age (days)", min_value=0, value=450)
            payment_day = st.slider("Typical Payment Day", 1, 28, 15)
            payment_hour = st.slider("Typical Payment Hour", 0, 23, 10)
            avg_amount = st.number_input("Avg Monthly Amount ($)", min_value=1.0, value=89.99)

        submitted = st.form_submit_button("🔍 Analyze & Decide", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "transaction_id": txn_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": currency,
            "decline_code": decline_code,
            "failure_reason": failure_reason or None,
            "past_success_count": past_success,
            "past_failure_count": past_failures,
            "account_age_days": account_age,
            "typical_payment_day": payment_day,
            "typical_payment_hour": payment_hour,
            "avg_amount": avg_amount,
        }

        with st.spinner("Running AI pipeline …"):
            try:
                r = requests.post(f"{API_BASE}/decide", json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()

                    st.success("✅ Decision complete!")
                    st.divider()

                    # Classification
                    cls = data["classification"]
                    st.subheader("1️⃣ Classification")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Category", cls["category"].replace("_", " ").title())
                    col2.metric("Confidence", f"{cls['confidence']:.1%}")
                    col3.metric("Method", cls["method"])
                    st.caption(f"Reasoning: {cls['reasoning']}")

                    st.divider()

                    # Decisions side by side
                    st.subheader("2️⃣ Recovery Decisions")
                    ai = data["ai_decision"]
                    bl = data["baseline_decision"]

                    col1, col2 = st.columns(2)
                    with col1:
                        action_emoji = {"retry": "🟢", "escalate": "🔴", "no_action": "🟡"}.get(ai["action"], "⚪")
                        st.markdown(f"**🤖 AI Policy** — {action_emoji} {ai['action'].upper()}")
                        if ai.get("recovery_probability") is not None:
                            st.metric("Recovery Probability", f"{ai['recovery_probability']:.1%}")
                        if ai.get("retry_after_hours"):
                            st.metric("Retry In", f"{ai['retry_after_hours']} hours")
                        st.metric("Channel", ai.get("retry_channel", "—"))
                        st.caption(ai["reasoning"])

                        if ai.get("scored_candidates"):
                            st.markdown("**All timing scores:**")
                            sc_df = pd.DataFrame(ai["scored_candidates"])
                            sc_df["probability"] = sc_df["probability"].apply(lambda x: f"{x:.1%}")
                            st.dataframe(sc_df, use_container_width=True, hide_index=True)

                    with col2:
                        action_emoji = {"retry": "🟢", "escalate": "🔴", "no_action": "🟡"}.get(bl["action"], "⚪")
                        st.markdown(f"**📏 Baseline Policy** — {action_emoji} {bl['action'].upper()}")
                        if bl.get("retry_after_hours"):
                            st.metric("First Retry At", f"{bl['retry_after_hours']} hours")
                        st.metric("Channel", bl.get("retry_channel", "—"))
                        st.caption(bl["reasoning"])

                    st.divider()

                    # Messages
                    st.subheader("3️⃣ Generated Recovery Message")
                    msg = data["email_message"]
                    tab1, tab2 = st.tabs(["📧 Email", "📱 SMS"])
                    with tab1:
                        st.markdown(f"**Subject:** {msg['email_subject']}")
                        st.text_area("Email Body", msg["email_body"], height=200, disabled=True)
                        st.caption(f"Generated by: {msg['generated_by']}")
                    with tab2:
                        st.text_area("SMS Body (max 160 chars)", msg["sms_body"], height=100, disabled=True)
                        chars = len(msg["sms_body"])
                        st.caption(f"{chars}/160 characters")

                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ FastAPI server not running. In a separate terminal, run:\n"
                    "```bash\nuvicorn api.main:app --reload\n```"
                )
