"""
ui/streamlit_app.py
---------------------
Streamlit front-end for the Financial News Analyst.

Run with:
    streamlit run ui/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial News Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .sentiment-bullish  { color: #2e7d32; font-weight: bold; font-size: 1.2rem; }
    .sentiment-bearish  { color: #c62828; font-weight: bold; font-size: 1.2rem; }
    .sentiment-neutral  { color: #f57c00; font-weight: bold; font-size: 1.2rem; }
    .sentiment-mixed    { color: #6a1b9a; font-weight: bold; font-size: 1.2rem; }
    .disclaimer-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #e65100;
        margin-top: 1rem;
    }
    .source-chip {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    use_api = st.checkbox("Use FastAPI backend", value=False,
                          help="If unchecked, calls the pipeline directly in-process.")
    st.session_state.setdefault("api_url", "http://localhost:8000/api/v1")
    api_url = st.text_input("API URL", key="api_url", disabled=not use_api)

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "Why did Tesla stock drop today?",
        "Summarize current market sentiment for S&P 500",
        "What's happening with Bitcoin prices?",
        "How is Nvidia performing this week?",
        "Give me an overview of AAPL and MSFT",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.markdown("### 📊 RAG Store")
    if st.button("🔄 Refresh stats"):
        try:
            from app.rag.vector_store import get_vector_store
            count = get_vector_store().count()
            st.success(f"📚 {count} documents in store")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Financial News Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Agent AI System · RAG-powered · Real-time data</div>',
            unsafe_allow_html=True)

# Query input
st.session_state.setdefault("query_input", "")
query = st.text_input(
    "Ask a financial question",
    placeholder="e.g. Why did Tesla stock drop today?",
    key="query_input",
    help="Max 500 characters. Prompt injection is blocked.",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col_clear:
    if st.button("✕ Clear"):
        st.session_state["query_input"] = ""
        st.rerun()


# ── Pipeline execution ────────────────────────────────────────────────────────
if analyze_clicked and query.strip():
    # Input validation
    try:
        from app.core.security import validate_query, ValidationError
        clean_query = validate_query(query)
    except Exception as ve:
        st.error(f"🚫 Input rejected: {ve}")
        st.stop()

    with st.spinner("🤖 Agents working… fetching data, news, and generating insight…"):
        t0 = time.perf_counter()
        state = None
        error = None

        if use_api:
            # ── Call FastAPI backend ──────────────────────────────────────────
            import requests as req
            try:
                resp = req.post(
                    f"{api_url}/analyze",
                    json={"query": clean_query},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                state = data
            except Exception as e:
                error = str(e)
        else:
            # ── In-process pipeline ───────────────────────────────────────────
            try:
                from app.agents.supervisor_agent import run_pipeline
                state = run_pipeline(clean_query)
            except Exception as e:
                error = str(e)

        elapsed = time.perf_counter() - t0

    if error:
        st.error(f"❌ Pipeline error: {error}")
        st.stop()

    if state:
        analysis = state.get("analysis", {})
        tickers = state.get("tickers", [])
        rag_sources = state.get("rag_sources", [])
        token_usage = state.get("token_usage", {})
        agent_log = state.get("agent_log", [])
        articles_count = state.get("articles_count") or len(state.get("articles", []))

        # ── Metrics row ───────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("⏱ Latency", f"{state.get('total_latency_s', round(elapsed,2))}s")
        with m2:
            st.metric("🪙 Tokens Used", token_usage.get("total", "—"))
        with m3:
            st.metric("📰 Articles", articles_count)
        with m4:
            st.metric("📚 RAG Docs", len(rag_sources))

        st.markdown("---")

        # ── Sentiment badge ───────────────────────────────────────────────────
        sentiment = analysis.get("sentiment", "Neutral")
        css_class = f"sentiment-{sentiment.lower()}"
        sentiment_icons = {
            "Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡", "Mixed": "🟣"
        }
        icon = sentiment_icons.get(sentiment, "⚪")
        st.markdown(
            f'<span class="{css_class}">{icon} {sentiment}</span>',
            unsafe_allow_html=True,
        )

        # ── Summary ───────────────────────────────────────────────────────────
        st.subheader("📋 Summary")
        st.write(analysis.get("summary", "No summary generated."))

        # ── Key drivers / risk factors ────────────────────────────────────────
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("🚀 Key Drivers")
            drivers = analysis.get("key_drivers", [])
            for d in drivers:
                st.markdown(f"• {d}")
            if not drivers:
                st.caption("None identified.")

        with col_right:
            st.subheader("⚠️ Risk Factors")
            risks = analysis.get("risk_factors", [])
            for r in risks:
                st.markdown(f"• {r}")
            if not risks:
                st.caption("None identified.")

        # ── Insight ───────────────────────────────────────────────────────────
        st.subheader("💡 Educational Insight")
        st.info(analysis.get("insight", "No insight generated."))

        # ── Tickers ───────────────────────────────────────────────────────────
        if tickers:
            st.subheader("📊 Tickers Analyzed")
            st.write(" · ".join(tickers))

        # ── RAG Sources ───────────────────────────────────────────────────────
        st.subheader("📚 Sources Used (RAG)")
        if rag_sources:
            chips = "".join(
                f'<span class="source-chip">{s}</span>' for s in rag_sources
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No RAG sources retrieved.")

        # Also show sources_used from LLM output if present
        llm_sources = analysis.get("sources_used", [])
        if llm_sources:
            with st.expander("LLM-cited sources"):
                for s in llm_sources:
                    st.markdown(f"• {s}")

        # ── Agent trace ───────────────────────────────────────────────────────
        with st.expander("🔎 Agent Execution Trace"):
            for step in agent_log:
                st.code(step, language=None)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="disclaimer-box">⚠️ '
            + analysis.get("disclaimer", "This is not financial advice.")
            + "</div>",
            unsafe_allow_html=True,
        )

elif analyze_clicked:
    st.warning("Please enter a question before clicking Analyze.")
