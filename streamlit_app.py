import streamlit as st
from main import run_agent

st.set_page_config(page_title="Financial Loan Agent", page_icon="💳", layout="centered")

st.markdown("""
<style>
.green-box { background-color: #0d3320; border-left: 6px solid #00e676; padding: 20px; border-radius: 8px; margin: 10px 0; }
.yellow-box { background-color: #332d00; border-left: 6px solid #ffd600; padding: 20px; border-radius: 8px; margin: 10px 0; }
.red-box { background-color: #330d0d; border-left: 6px solid #ff1744; padding: 20px; border-radius: 8px; margin: 10px 0; }
.metric-card { background-color: #1a1a2e; padding: 15px; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("💳 Agentic Financial Loan Advisor")
st.markdown("*Powered by LangGraph + ChromaDB + Gemini 2.5 Flash*")
st.divider()

# Quick scenario presets
st.subheader("⚡ Quick Scenarios")
col1, col2, col3, col4 = st.columns(4)
preset = None
with col1:
    if st.button("✅ Strong Applicant"):
        preset = (800, 250000, 100000, "employed for 5 years, no debt")
with col2:
    if st.button("⚠️ Borderline"):
        preset = (660, 80000, 50000, "first time borrower")
with col3:
    if st.button("❌ High Risk"):
        preset = (550, 30000, 100000, "recently unemployed")
with col4:
    if st.button("🏦 Premium Loan"):
        preset = (820, 500000, 400000, "self-employed, 3 years tax returns available")

st.divider()
st.subheader("📋 Applicant Details")

# Pre-fill if preset selected
default_credit = preset[0] if preset else 650
default_income = preset[1] if preset else 50000
default_loan = preset[2] if preset else 100000
default_info = preset[3] if preset else ""

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=default_credit)
with col2:
    annual_income = st.number_input("Annual Income (USD)", min_value=0, value=default_income, step=5000)
with col3:
    loan_amount = st.number_input("Loan Amount (USD)", min_value=1000, value=default_loan, step=5000)

extra_info = st.text_area("Additional Info (optional)", value=default_info,
    placeholder="e.g. first time borrower, has bankruptcy history, self-employed, co-signer available...")

# Credit score indicator
if credit_score >= 800:
    st.success(f"Credit Score: {credit_score} — Exceptional 🌟")
elif credit_score >= 740:
    st.success(f"Credit Score: {credit_score} — Very Good ✅")
elif credit_score >= 670:
    st.info(f"Credit Score: {credit_score} — Good 👍")
elif credit_score >= 580:
    st.warning(f"Credit Score: {credit_score} — Fair ⚠️")
else:
    st.error(f"Credit Score: {credit_score} — Poor ❌")

dti = round((loan_amount / annual_income) * 100, 1) if annual_income > 0 else 0
st.caption(f"Loan-to-Income Ratio: {dti}% {'✅' if dti <= 250 else '⚠️'}")

if st.button("🚀 Analyze Application", type="primary", use_container_width=True):
    query = f"Applicant has a credit score of {credit_score}, annual income of {annual_income} USD, and is requesting a loan of {loan_amount} USD."
    if extra_info:
        query += f" Additional info: {extra_info}"

    with st.spinner("🤖 Agent analyzing..."):
        result = run_agent(query)

    st.divider()
    st.subheader("📊 Analysis Results")

    decision = result["decision"]
    if decision == "GREEN":
        st.markdown('<div class="green-box"><h3>✅ APPROVED — GREEN</h3></div>', unsafe_allow_html=True)
    elif decision == "YELLOW":
        st.markdown('<div class="yellow-box"><h3>⚠️ MANUAL REVIEW — YELLOW</h3></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="red-box"><h3>❌ REJECTED — RED</h3></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📝 Full Agent Analysis")
    st.write(result["full_response"])

    with st.expander("📚 Guidelines Retrieved from ChromaDB"):
        for i, doc in enumerate(result["retrieved_docs"]):
            st.write(f"**{i+1}.** {doc}")

    with st.expander("🔍 Agent Reasoning"):
        st.write(result["analysis"])

st.divider()
st.caption("Built with LangGraph + ChromaDB + Gemini 2.5 Flash | Agentic AI Demo")
