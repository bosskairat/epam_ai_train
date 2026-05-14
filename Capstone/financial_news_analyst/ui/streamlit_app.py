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

import requests
import streamlit as st
import time

API_BASE = "http://localhost:8000/api/v1"

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
    .history-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .history-meta {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.25rem;
    }
    .history-query {
        font-size: 1rem;
        font-weight: 600;
        color: #222;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Example Queries")
    examples = [
        "Why did Tesla stock drop today?",
        "Summarize current market sentiment for S&P 500",
        "What's happening with Bitcoin prices?",
        "How is Nvidia performing this week?",
        "Give me an overview of AAPL and MSFT",
    ]
    for ex in examples:
        st.button(
            ex, key=ex, use_container_width=True,
            on_click=lambda q=ex: st.session_state.update({"query_input": q}),
        )

    st.markdown("---")
    st.markdown("### 📊 RAG Store")
    if st.button("🔄 Refresh stats"):
        try:
            resp = requests.get(f"{API_BASE}/rag/stats", timeout=5)
            resp.raise_for_status()
            count = resp.json().get("document_count", "?")
            st.success(f"📚 {count} documents in store")
        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Financial News Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Agent AI System · RAG-powered · Real-time data</div>',
            unsafe_allow_html=True)


def _md(text: str) -> str:
    """Escape $ so Streamlit doesn't treat them as LaTeX delimiters."""
    return str(text).replace("$", r"\$")


def _sentiment_html(sentiment: str) -> str:
    icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡", "Mixed": "🟣"}
    css = f"sentiment-{sentiment.lower()}"
    icon = icons.get(sentiment, "⚪")
    return f'<span class="{css}">{icon} {sentiment}</span>'


def _render_analysis(state: dict, elapsed: float | None = None, nested: bool = False) -> None:
    """Render a pipeline result (used in both Analyze and History tabs).

    nested=True avoids st.expander calls (Streamlit forbids nested expanders).
    """
    analysis   = state.get("analysis", {})
    tickers    = state.get("tickers", [])
    rag_sources = state.get("rag_sources", [])
    token_usage = state.get("token_usage", {})
    agent_log  = state.get("agent_log", [])
    articles_count = state.get("articles_count") or len(state.get("articles", []))
    latency    = state.get("total_latency_s", round(elapsed, 2) if elapsed else "—")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("⏱ Latency", f"{latency}s")
    with m2:
        st.metric("🪙 Tokens Used", token_usage.get("total", "—"))
    with m3:
        st.metric("📰 Articles", articles_count)
    with m4:
        st.metric("📚 RAG Docs", len(rag_sources))

    st.markdown("---")

    # Sentiment badge
    sentiment = analysis.get("sentiment", "Neutral")
    st.markdown(_sentiment_html(sentiment), unsafe_allow_html=True)

    # Summary
    st.subheader("📋 Summary")
    st.markdown(_md(analysis.get("summary", "No summary generated.")))

    # Key drivers / risk factors
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🚀 Key Drivers")
        for d in analysis.get("key_drivers", []):
            st.markdown(f"• {_md(d)}")
        if not analysis.get("key_drivers"):
            st.caption("None identified.")

    with col_right:
        st.subheader("⚠️ Risk Factors")
        for r in analysis.get("risk_factors", []):
            st.markdown(f"• {_md(r)}")
        if not analysis.get("risk_factors"):
            st.caption("None identified.")

    # Insight
    st.subheader("💡 Educational Insight")
    st.info(_md(analysis.get("insight", "No insight generated.")))

    # Tickers
    if tickers:
        st.subheader("📊 Tickers Analyzed")
        st.write(" · ".join(tickers))

    # RAG Sources
    st.subheader("📚 Sources Used (RAG)")
    if rag_sources:
        chips = "".join(f'<span class="source-chip">{s}</span>' for s in rag_sources)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No RAG sources retrieved.")

    llm_sources = analysis.get("sources_used", [])
    if llm_sources:
        if nested:
            st.markdown("**LLM-cited sources**")
            for s in llm_sources:
                if str(s).startswith("http"):
                    st.markdown(f"• [{s}]({s})")
                else:
                    st.markdown(f"• {s}")
        else:
            with st.expander("LLM-cited sources"):
                for s in llm_sources:
                    if str(s).startswith("http"):
                        st.markdown(f"• [{s}]({s})")
                    else:
                        st.markdown(f"• {s}")

    # Agent trace
    if nested:
        if agent_log:
            st.markdown("**🔎 Agent Execution Trace**")
            for step in agent_log:
                st.code(step, language=None)
    else:
        with st.expander("🔎 Agent Execution Trace"):
            for step in agent_log:
                st.code(step, language=None)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer-box">⚠️ '
        + analysis.get("disclaimer", "This is not financial advice.")
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analyze, tab_history = st.tabs(["🔍 Analyze", "📜 History"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyze
# ════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.session_state.setdefault("query_input", "")
    query = st.text_input(
        "Ask a financial question",
        placeholder="e.g. Why did Tesla stock drop today?",
        key="query_input",
        help="Max 500 characters. Prompt injection is blocked.",
    )

    def _clear_query():
        st.session_state["query_input"] = ""

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col_clear:
        st.button("✕ Clear", on_click=_clear_query)

    if analyze_clicked and query.strip():
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

            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"query": clean_query},
                    timeout=120,
                )
                resp.raise_for_status()
                state = resp.json()
            except requests.exceptions.ConnectionError:
                error = (
                    "Cannot connect to the FastAPI server. "
                    "Start it with:  uvicorn main:app --reload"
                )
            except requests.exceptions.HTTPError as e:
                error = f"Server error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                error = str(e)

            elapsed = time.perf_counter() - t0

        if error:
            st.error(f"❌ {error}")
            st.stop()

        if state:
            _render_analysis(state, elapsed)

    elif analyze_clicked:
        st.warning("Please enter a question before clicking Analyze.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — History
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    col_refresh, col_clear_hist, col_spacer = st.columns([1, 1, 6])
    with col_refresh:
        refresh_hist = st.button("🔄 Refresh", key="hist_refresh", use_container_width=True)
    with col_clear_hist:
        clear_hist = st.button("🗑 Clear All", key="hist_clear", use_container_width=True)

    if clear_hist:
        try:
            resp = requests.delete(f"{API_BASE}/history", timeout=10)
            resp.raise_for_status()
            deleted = resp.json().get("deleted", "?")
            st.success(f"Deleted {deleted} record(s).")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the FastAPI server.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Load history
    history_data = []
    try:
        resp = requests.get(f"{API_BASE}/history?limit=100", timeout=10)
        resp.raise_for_status()
        history_data = resp.json().get("conversations", [])
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the FastAPI server.")
    except Exception as e:
        st.error(f"Could not load history: {e}")

    if not history_data:
        st.info("No conversation history yet. Run an analysis first.")
    else:
        st.markdown(f"**{len(history_data)} past conversation(s)**")
        sentiment_icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡", "Mixed": "🟣"}

        for record in history_data:
            created = record.get("created_at", "")[:19].replace("T", " ")
            sentiment = record.get("sentiment", "Neutral")
            icon = sentiment_icons.get(sentiment, "⚪")
            tickers_str = ", ".join(record.get("tickers", [])) or "—"
            tokens = record.get("token_total", 0)
            latency = record.get("latency_s", 0.0)

            label = f"{icon} {record['query'][:80]}{'…' if len(record['query']) > 80 else ''}"
            with st.expander(label):
                st.markdown(
                    f'<div class="history-meta">'
                    f'🕐 {created} &nbsp;|&nbsp; '
                    f'📊 {tickers_str} &nbsp;|&nbsp; '
                    f'🪙 {tokens} tokens &nbsp;|&nbsp; '
                    f'⏱ {latency}s'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                full = record.get("full_state", {})
                if full:
                    _render_analysis(full, nested=True)
                else:
                    # Fallback: minimal display from summary columns
                    st.markdown(_md(record.get("summary", "No summary.")))
                    rag = record.get("rag_sources", [])
                    if rag:
                        chips = "".join(
                            f'<span class="source-chip">{s}</span>' for s in rag
                        )
                        st.markdown(chips, unsafe_allow_html=True)
