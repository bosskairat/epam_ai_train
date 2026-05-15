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
from streamlit_cookies_controller import CookieController

_COOKIE_NAME = "fin_analyst_token"

API_BASE = "http://localhost:8000/api/v1"

# ── Page config (must be first Streamlit call) ────────────────────────────────
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
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 3px 4px;
        white-space: nowrap;
        vertical-align: middle;
        line-height: 1.4;
    }
    .history-meta {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cookie controller (must be after set_page_config) ────────────────────────
_cookie = CookieController()

# ── Header rendered immediately — visible before any network call ─────────────
st.markdown('<div class="main-header">📈 Financial News Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Agent AI System · RAG-powered · Real-time data</div>',
            unsafe_allow_html=True)


# ── Restore session from cookie on page refresh ───────────────────────────────
if not st.session_state.get("token"):
    _saved = _cookie.get(_COOKIE_NAME)
    if _saved:
        with st.spinner("Restoring session…"):
            try:
                _resp = requests.get(
                    f"{API_BASE}/auth/me",
                    headers={"Authorization": f"Bearer {_saved}"},
                    timeout=5,
                )
                if _resp.status_code == 200:
                    _me = _resp.json()
                    st.session_state["token"] = _saved
                    st.session_state["logged_in_user"] = _me["username"]
                    st.session_state["is_admin"] = _me.get("is_admin", False)
                    st.rerun()
                else:
                    _cookie.remove(_COOKIE_NAME)
            except Exception:
                _cookie.remove(_COOKIE_NAME)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _get_auth_config() -> dict:
    """Fetch auth config once per browser session (cached in session state)."""
    if "auth_config" not in st.session_state:
        try:
            r = requests.get(f"{API_BASE}/auth/config", timeout=5)
            r.raise_for_status()
            st.session_state["auth_config"] = r.json()
        except Exception:
            st.session_state["auth_config"] = {
                "auth_required": False,
                "registration_enabled": False,
            }
    return st.session_state["auth_config"]


def _auth_headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _store_session(token: str, username: str, is_admin: bool = False) -> None:
    st.session_state["token"] = token
    st.session_state["logged_in_user"] = username
    st.session_state["is_admin"] = is_admin
    _cookie.set(_COOKIE_NAME, token)


def _do_login(username: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            _store_session(data["access_token"], data["username"], data.get("is_admin", False))
            return True
        st.session_state["_auth_error"] = resp.json().get("detail", "Login failed.")
        return False
    except requests.exceptions.ConnectionError:
        st.session_state["_auth_error"] = "Cannot connect to the server."
        return False
    except Exception as exc:
        st.session_state["_auth_error"] = str(exc)
        return False


def _do_register(username: str, password: str) -> bool:
    """Register and auto-login if the server returns a token."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/register",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("access_token"):
                _store_session(data["access_token"], data["username"], data.get("is_admin", False))
            return True
        st.session_state["_auth_error"] = resp.json().get("detail", "Registration failed.")
        return False
    except requests.exceptions.ConnectionError:
        st.session_state["_auth_error"] = "Cannot connect to the server."
        return False
    except Exception as exc:
        st.session_state["_auth_error"] = str(exc)
        return False


def _fetch_token_usage() -> int:
    """Return the number of tokens the current user has spent."""
    try:
        resp = requests.get(f"{API_BASE}/usage", headers=_auth_headers(), timeout=5)
        if resp.status_code == 200:
            return resp.json().get("tokens_used", 0)
    except Exception:
        pass
    return 0


# ── Auth config (session-state cached — no blocking call on reruns) ───────────
auth_cfg = _get_auth_config()
registration_enabled: bool = auth_cfg.get("registration_enabled", False)


# ── Login / Register wall ─────────────────────────────────────────────────────
# On first render after refresh the cookie component hasn't sent values yet.
# Use a flag so we wait one render cycle before showing the login form.
_cookie_checked = st.session_state.get("_cookie_checked", False)
if not _cookie_checked:
    st.session_state["_cookie_checked"] = True
    st.rerun()  # give the cookie component one cycle to load browser cookies

if not st.session_state.get("token"):
    st.markdown("---")

    # Narrow the form by placing it in the centre column
    _, form_col, _ = st.columns([3, 2, 3])

    with form_col:
        tab_login, tab_reg = (
            st.tabs(["🔑 Login", "📝 Register"])
            if registration_enabled
            else (st.container(), None)
        )

        # ── Login tab ─────────────────────────────────────────────────────────
        with tab_login:
            if not registration_enabled:
                st.markdown("### 🔑 Login")
            with st.form("login_form"):
                username_in = st.text_input("Username", key="li_user")
                password_in = st.text_input("Password", type="password", key="li_pass")
                login_btn = st.form_submit_button("Login", type="primary", use_container_width=True)

            if st.session_state.get("_auth_error") and not registration_enabled:
                st.error(st.session_state.pop("_auth_error"))

            if login_btn:
                if not username_in or not password_in:
                    st.error("Enter username and password.")
                else:
                    with st.spinner("Logging in…"):
                        ok = _do_login(username_in, password_in)
                    if ok:
                        st.rerun()
                    else:
                        st.error(st.session_state.pop("_auth_error", "Login failed."))

        # ── Register tab ──────────────────────────────────────────────────────
        if tab_reg is not None:
            with tab_reg:
                with st.form("register_form"):
                    reg_user = st.text_input(
                        "Username", help="3–64 characters: letters, digits, _ . -", key="reg_user"
                    )
                    reg_pass = st.text_input("Password", type="password",
                                             help="Minimum 8 characters", key="reg_pass")
                    reg_pass2 = st.text_input("Confirm password", type="password", key="reg_pass2")
                    reg_btn = st.form_submit_button(
                        "Create account", type="primary", use_container_width=True
                    )

                if reg_btn:
                    if not reg_user or not reg_pass:
                        st.error("Fill in all fields.")
                    elif reg_pass != reg_pass2:
                        st.error("Passwords do not match.")
                    elif len(reg_pass) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        with st.spinner("Creating account…"):
                            ok = _do_register(reg_user, reg_pass)
                        if ok:
                            if st.session_state.get("token"):
                                st.rerun()
                            else:
                                st.success("Account created! Switch to the Login tab.")
                        else:
                            st.error(st.session_state.pop("_auth_error", "Registration failed."))

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Authenticated: route to admin or user page
# ─────────────────────────────────────────────────────────────────────────────

def _logout():
    for k in ("token", "logged_in_user", "is_admin", "history_data",
              "history_loaded", "auth_config"):
        st.session_state.pop(k, None)
    _cookie.remove(_COOKIE_NAME)


# ── Shared sidebar header (username + logout) ─────────────────────────────────
with st.sidebar:
    if st.session_state.get("logged_in_user"):
        role = "👑 Admin" if st.session_state.get("is_admin") else "👤"
        st.markdown(f"{role} **{st.session_state['logged_in_user']}**")
        if st.button("Logout", use_container_width=True):
            _logout()
            st.rerun()
        st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# ADMIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.get("is_admin"):

    st.markdown("## 👑 Admin Panel")
    admin_tab_users, admin_tab_rag = st.tabs(["👥 User Management", "🔬 RAG Evaluation"])

    with admin_tab_rag:
        st.markdown("### RAG Quality Metrics — Aggregate over History")
        sample = st.slider("Conversations to evaluate", 5, 50, 20, key="eval_sample")
        if st.button("▶ Run Evaluation", key="run_rag_eval", type="primary"):
            with st.spinner("Evaluating RAG quality… this may take ~30 s"):
                try:
                    ev_resp = requests.get(
                        f"{API_BASE}/rag/evaluate/history?sample={sample}",
                        headers=_auth_headers(), timeout=120,
                    )
                    if ev_resp.ok:
                        st.session_state["admin_rag_eval"] = ev_resp.json()
                    else:
                        st.error(ev_resp.text)
                except Exception as e:
                    st.error(str(e))

        ev = st.session_state.get("admin_rag_eval")
        if ev:
            if "error" in ev:
                st.warning(ev["error"])
            else:
                st.caption(f"Sample: **{ev.get('sample_size', 0)}** conversations")
                st.markdown("---")

                def _agg_row(label: str, data: dict, fmt=".3f", invert=False):
                    m = data.get("mean")
                    if m is None:
                        st.markdown(f"**{label}**: no data")
                        return
                    icon = ("🟢" if m >= 0.6 else ("🟡" if m >= 0.35 else "🔴"))
                    if invert:
                        icon = ("🔴" if m >= 0.6 else ("🟡" if m >= 0.35 else "🟢"))
                    n = data.get("n", 0)
                    st.metric(
                        label,
                        f"{icon} {m:{fmt}}",
                        f"min {data.get('min', 0):{fmt}} / max {data.get('max', 0):{fmt}}  (n={n})",
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    _agg_row("📥 Retrieval Precision@K", ev.get("retrieval_precision", {}), ".2%")
                with c2:
                    _agg_row("💬 Answer Relevance (cosine)", ev.get("answer_relevance", {}))
                with c3:
                    _agg_row("🔗 Source Traceability", ev.get("source_attribution", {}))
                with c4:
                    _agg_row("🧠 Hallucination Score", ev.get("hallucination", {}), invert=True)

                st.markdown("---")
                st.markdown("**⚖️ Sentiment Distribution (historical)**")
                dist = ev.get("sentiment_distribution", {})
                if dist:
                    icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡", "Mixed": "🟣"}
                    dcols = st.columns(len(dist))
                    for i, (sent, pct) in enumerate(sorted(dist.items())):
                        dcols[i].metric(
                            f"{icons.get(sent, '⚪')} {sent}",
                            f"{pct:.0%}",
                        )
                    # Bias flag
                    for sent, pct in dist.items():
                        if pct > 0.70:
                            st.warning(f"⚠️ Potential bias: **{sent}** dominates {pct:.0%} of responses.")

    with admin_tab_users:
        def _load_users() -> list:
            try:
                resp = requests.get(
                    f"{API_BASE}/admin/users", headers=_auth_headers(), timeout=10
                )
                resp.raise_for_status()
                return resp.json().get("users", [])
            except Exception as e:
                st.error(f"Failed to load users: {e}")
                return []

        if st.button("🔄 Refresh", key="admin_refresh"):
            st.session_state.pop("admin_users", None)

        if "admin_users" not in st.session_state:
            with st.spinner("Loading users…"):
                st.session_state["admin_users"] = _load_users()

        users: list = st.session_state.get("admin_users", [])

        # ── Summary metrics ───────────────────────────────────────────────────
        total = len(users)
        blocked_count = sum(1 for u in users if u["is_blocked"])
        total_tokens = sum(u.get("tokens_used", 0) for u in users)

        rag_data = {}
        try:
            rag_resp = requests.get(f"{API_BASE}/rag/stats", headers=_auth_headers(), timeout=5)
            if rag_resp.ok:
                rag_data = rag_resp.json()
        except Exception:
            pass
        rag_count = rag_data.get("document_count", "?")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Total Users", total)
        c2.metric("🚫 Blocked", blocked_count)
        c3.metric("🪙 Total Tokens Used", f"{total_tokens:,}")
        c4.metric("📚 RAG Documents", rag_count)
        st.markdown("---")

        # ── Users table ───────────────────────────────────────────────────────
        if not users:
            st.info("No users found.")
        else:
            me = st.session_state.get("logged_in_user", "")

            for u in users:
                uname = u["username"]
                created = u.get("created_at", "")[:10]
                tokens = u.get("tokens_used", 0)
                blocked = u["is_blocked"]

                status_badge = "🚫 Blocked" if blocked else "✅ Active"
                is_me = uname == me

                with st.expander(f"{status_badge}  **{uname}**  —  registered {created}  |  🪙 {tokens:,} tokens"):
                    col_info, col_actions = st.columns([3, 2])

                    with col_info:
                        st.markdown(f"**Username:** `{uname}`")
                        st.markdown(f"**Registered:** {u.get('created_at', '')[:19].replace('T', ' ')}")
                        st.markdown(f"**Tokens used:** {tokens:,}")
                        st.markdown(f"**Status:** {status_badge}")
                        if is_me:
                            st.info("This is your account.")

                    with col_actions:
                        if not is_me:
                            if blocked:
                                if st.button("✅ Unblock", key=f"unblock_{uname}", use_container_width=True):
                                    try:
                                        r = requests.patch(
                                            f"{API_BASE}/admin/users/{uname}/unblock",
                                            headers=_auth_headers(), timeout=10,
                                        )
                                        r.raise_for_status()
                                        st.success(f"{uname} unblocked.")
                                        st.session_state.pop("admin_users", None)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                            else:
                                if st.button("🚫 Block", key=f"block_{uname}", use_container_width=True):
                                    try:
                                        r = requests.patch(
                                            f"{API_BASE}/admin/users/{uname}/block",
                                            headers=_auth_headers(), timeout=10,
                                        )
                                        r.raise_for_status()
                                        st.success(f"{uname} blocked.")
                                        st.session_state.pop("admin_users", None)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))

                            st.markdown("---")

                            if st.checkbox("🔑 Reset password", key=f"show_pw_{uname}"):
                                new_pw = st.text_input(
                                    "New password (min 8 chars)", type="password",
                                    key=f"pw_{uname}",
                                )
                                if st.button("Set password", key=f"setpw_{uname}"):
                                    if len(new_pw) < 8:
                                        st.error("Password must be at least 8 characters.")
                                    else:
                                        try:
                                            r = requests.post(
                                                f"{API_BASE}/admin/users/{uname}/reset-password",
                                                headers=_auth_headers(),
                                                json={"new_password": new_pw},
                                                timeout=10,
                                            )
                                            r.raise_for_status()
                                            st.success("Password updated.")
                                        except Exception as e:
                                            st.error(str(e))

                            if st.checkbox("🗑 Delete account", key=f"show_del_{uname}"):
                                st.warning(f"Permanently delete **{uname}**? This cannot be undone.")
                                if st.button("Confirm delete", key=f"del_{uname}", type="primary"):
                                    try:
                                        r = requests.delete(
                                            f"{API_BASE}/admin/users/{uname}",
                                            headers=_auth_headers(), timeout=10,
                                        )
                                        r.raise_for_status()
                                        st.success(f"{uname} deleted.")
                                        st.session_state.pop("admin_users", None)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))


# ═════════════════════════════════════════════════════════════════════════════
# USER PAGE — Financial Analysis
# ═════════════════════════════════════════════════════════════════════════════
else:

    # ── Sidebar (user-specific) ───────────────────────────────────────────────
    with st.sidebar:
        tokens_used = _fetch_token_usage()
        st.caption(f"🪙 Tokens used: **{tokens_used:,}**")

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
                ex, key=f"ex|{ex}", use_container_width=True,
                on_click=lambda q=ex: st.session_state.update({"query_input": q}),
            )

        st.markdown("---")
        st.caption("⚠️ For educational purposes only. Not financial advice.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _md(text: str) -> str:
        return str(text).replace("$", r"\$")

    def _sentiment_html(sentiment: str) -> str:
        icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡", "Mixed": "🟣"}
        return f'<span class="sentiment-{sentiment.lower()}">{icons.get(sentiment, "⚪")} {sentiment}</span>'

    def _render_analysis(state: dict, elapsed=None, nested: bool = False) -> None:
        analysis       = state.get("analysis", {})
        tickers        = state.get("tickers", [])
        rag_sources    = state.get("rag_sources", [])
        token_usage    = state.get("token_usage", {})
        agent_log      = state.get("agent_log", [])
        articles_count = state.get("articles_count") or len(state.get("articles", []))
        latency        = state.get("total_latency_s", round(elapsed, 2) if elapsed else "—")
        hallucination  = state.get("hallucination_score")

        cols = st.columns(5)
        cols[0].metric("⏱ Latency", f"{latency}s")
        cols[1].metric("🪙 Tokens", token_usage.get("total", "—"))
        cols[2].metric("📰 Articles", articles_count)
        cols[3].metric("📚 RAG Docs", len(rag_sources))
        if hallucination is not None:
            color = "🟢" if hallucination < 0.3 else ("🟡" if hallucination < 0.6 else "🔴")
            cols[4].metric("🧠 Hallucination", f"{color} {hallucination:.2f}")

        st.markdown("---")
        st.markdown(_sentiment_html(analysis.get("sentiment", "Neutral")), unsafe_allow_html=True)

        st.subheader("📋 Summary")
        st.markdown(_md(analysis.get("summary", "No summary generated.")))

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

        st.subheader("💡 Educational Insight")
        st.info(_md(analysis.get("insight", "No insight generated.")))

        if tickers:
            st.subheader("📊 Tickers Analyzed")
            st.write(" · ".join(tickers))

        st.subheader("📚 Sources Used (RAG)")
        if rag_sources:
            st.markdown(
                "".join(f'<span class="source-chip">{s}</span>' for s in rag_sources),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No RAG sources retrieved.")

        llm_sources = analysis.get("sources_used", [])
        if llm_sources:
            if nested:
                st.markdown("**LLM-cited sources**")
                for s in llm_sources:
                    st.markdown(f"• [{s}]({s})" if str(s).startswith("http") else f"• {s}")
            else:
                with st.expander("LLM-cited sources"):
                    for s in llm_sources:
                        st.markdown(f"• [{s}]({s})" if str(s).startswith("http") else f"• {s}")

        if nested:
            if agent_log:
                st.markdown("**🔎 Agent Execution Trace**")
                for step in agent_log:
                    st.code(step, language=None)
        else:
            with st.expander("🔎 Agent Execution Trace"):
                for step in agent_log:
                    st.code(step, language=None)

        st.markdown(
            '<div class="disclaimer-box">⚠️ '
            + analysis.get("disclaimer", "This is not financial advice.")
            + "</div>",
            unsafe_allow_html=True,
        )

    def _render_rag_eval(ev: dict | None) -> None:
        if not ev:
            return
        with st.expander("🔬 RAG Quality Evaluation", expanded=False):
            composite = ev.get("composite_score")
            if composite is not None:
                color = "🟢" if composite >= 0.65 else ("🟡" if composite >= 0.4 else "🔴")
                st.markdown(f"**Composite Quality Score: {color} {composite:.2f}**")
                st.progress(float(composite))
                st.markdown("---")

            cols = st.columns(5)

            # 1 — Retrieval
            ret = ev.get("retrieval", {})
            with cols[0]:
                st.markdown("**📥 Retrieval**")
                st.metric("Precision@K", f"{ret.get('precision_at_k', 0):.0%}")
                st.metric("Mean Score",  f"{ret.get('mean_score', 0):.3f}")
                st.metric("MRR",         f"{ret.get('mrr', 0):.3f}")
                st.metric("Docs",        ret.get("doc_count", 0))

            # 2 — Answer relevance
            rel = ev.get("answer_relevance", {})
            with cols[1]:
                st.markdown("**💬 Relevance**")
                label_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                    rel.get("relevance_label", ""), "⚪"
                )
                st.metric("Cosine Sim",    f"{rel.get('cosine_similarity', 0):.3f}")
                st.metric("Token Overlap", f"{rel.get('token_overlap', 0):.0%}")
                st.markdown(f"Label: {label_icon} **{rel.get('relevance_label', '—')}**")

            # 3 — Source attribution
            attr = ev.get("source_attribution", {})
            with cols[2]:
                st.markdown("**🔗 Attribution**")
                st.metric("Traceability", f"{attr.get('traceability_score', 0):.2f}")
                st.metric("Cited URLs",   attr.get("llm_cited_urls_count", 0))
                st.metric("Coverage",     f"{attr.get('source_coverage', 0):.0%}")

            # 4 — Hallucination
            hall = ev.get("hallucination", {})
            with cols[3]:
                st.markdown("**🧠 Hallucination**")
                h_score = hall.get("hallucination_score", 1.0)
                h_icon  = "🟢" if h_score < 0.3 else ("🟡" if h_score < 0.6 else "🔴")
                st.metric("Score", f"{h_icon} {h_score:.2f}")
                st.metric("Claims", hall.get("claims_total", 0))
                st.metric("Supported", hall.get("claims_supported", 0))

            # 5 — Bias
            bias = ev.get("bias", {})
            lex  = bias.get("lexical", {})
            with cols[4]:
                st.markdown("**⚖️ Bias**")
                imb   = lex.get("imbalance_score", 0.0)
                b_icon = "🟢" if imb < 0.3 else ("🟡" if imb < 0.6 else "🔴")
                st.metric("Imbalance", f"{b_icon} {imb:.2f}")
                st.metric("🐂 Bullish KW", lex.get("bullish_keywords", 0))
                st.metric("🐻 Bearish KW", lex.get("bearish_keywords", 0))
                if bias.get("bias_flag"):
                    st.warning(bias.get("bias_reason", "Bias detected"))

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_analyze, tab_history = st.tabs(["🔍 Analyze", "📜 History"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Analyze
    # ════════════════════════════════════════════════════════════════════════
    with tab_analyze:
        st.session_state.setdefault("query_input", "")
        query = st.text_input(
            "Ask a financial question",
            placeholder="e.g. Why did Tesla stock drop today?",
            key="query_input",
            help="Max 500 characters. Prompt injection is blocked.",
        )
        consent = st.checkbox("Consent to store raw conversation", key="consent")

        def _clear_query():
            st.session_state["query_input"] = ""

        col_btn, col_clear = st.columns([1, 5])
        with col_btn:
            analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col_clear:
            st.button("✕ Clear", on_click=_clear_query)

        # Show cached result from previous run (after st.rerun())
        if not analyze_clicked and st.session_state.get("last_analysis"):
            _render_analysis(
                st.session_state["last_analysis"],
                st.session_state.get("last_analysis_elapsed"),
            )
            _render_rag_eval(st.session_state.get("last_rag_eval"))

        if analyze_clicked and query.strip():
            st.session_state.pop("last_analysis", None)
            st.session_state.pop("last_analysis_elapsed", None)
            try:
                from app.core.security import validate_query
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
                        headers=_auth_headers(),
                        json={"query": clean_query, "consent": bool(st.session_state.get("consent"))},
                        timeout=120,
                    )
                    if resp.status_code == 401:
                        for k in ("token", "logged_in_user", "is_admin"):
                            st.session_state.pop(k, None)
                        error = "Session expired. Please log in again."
                    else:
                        resp.raise_for_status()
                        state = resp.json()
                except requests.exceptions.ConnectionError:
                    error = "Cannot connect to the FastAPI server. Start it with: uvicorn main:app --reload"
                except requests.exceptions.HTTPError as e:
                    error = f"Server error {e.response.status_code}: {e.response.text}"
                except Exception as e:
                    error = str(e)
                elapsed = time.perf_counter() - t0

            if error:
                st.error(f"❌ {error}")
                if "Session expired" in (error or ""):
                    st.rerun()
                st.stop()

            if state:
                st.session_state.pop("history_loaded", None)
                st.session_state["last_analysis"] = state
                st.session_state["last_analysis_elapsed"] = elapsed
                # Run RAG evaluation in background (best-effort)
                try:
                    ev_resp = requests.post(
                        f"{API_BASE}/rag/evaluate",
                        headers=_auth_headers(),
                        json={"query": clean_query, "state": state},
                        timeout=30,
                    )
                    if ev_resp.ok:
                        st.session_state["last_rag_eval"] = ev_resp.json()
                except Exception:
                    pass
                st.rerun()  # rerender sidebar with updated token count

        elif analyze_clicked:
            st.warning("Please enter a question before clicking Analyze.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — History
    # ════════════════════════════════════════════════════════════════════════
    with tab_history:
        col_refresh, col_clear_hist, _ = st.columns([1, 1, 6])
        with col_refresh:
            refresh_hist = st.button("🔄 Refresh", key="hist_refresh", use_container_width=True)
        with col_clear_hist:
            clear_hist = st.button("🗑 Clear All", key="hist_clear", use_container_width=True)

        if clear_hist:
            try:
                resp = requests.delete(f"{API_BASE}/history", headers=_auth_headers(), timeout=10)
                resp.raise_for_status()
                deleted = resp.json().get("deleted", "?")
                st.success(f"Deleted {deleted} record(s).")
                st.session_state["history_data"] = []
                st.session_state["history_loaded"] = True
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FastAPI server.")
            except Exception as e:
                st.error(f"Error: {e}")

        if refresh_hist or not st.session_state.get("history_loaded"):
            with st.spinner("Loading history…"):
                try:
                    resp = requests.get(
                        f"{API_BASE}/history?limit=100", headers=_auth_headers(), timeout=10
                    )
                    resp.raise_for_status()
                    st.session_state["history_data"] = resp.json().get("conversations", [])
                    st.session_state["history_loaded"] = True
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the FastAPI server.")
                except Exception as e:
                    st.error(f"Could not load history: {e}")

        history_data: list = st.session_state.get("history_data", [])

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
                        st.markdown(_md(record.get("summary", "No summary.")))
                        rag = record.get("rag_sources", [])
                        if rag:
                            st.markdown(
                                "".join(f'<span class="source-chip">{s}</span>' for s in rag),
                                unsafe_allow_html=True,
                            )

                    # ── Feedback ───────────────────────────────────────────────
                    existing_rating   = record.get("user_rating")
                    existing_feedback = record.get("feedback_text") or ""
                    col_r, col_f, col_s = st.columns([1, 6, 1])
                    with col_r:
                        opts = ["No change", "1", "2", "3", "4", "5"]
                        default = str(existing_rating) if existing_rating else "No change"
                        rating_sel = st.selectbox(
                            "Rating", opts,
                            index=opts.index(default) if default in opts else 0,
                            key=f"rating_{record.get('id')}",
                        )
                    with col_f:
                        feedback_text = st.text_area(
                            "Feedback", value=existing_feedback,
                            key=f"feedback_{record.get('id')}", height=80,
                        )
                    with col_s:
                        if st.button("Submit", key=f"fb_{record.get('id')}"):
                            payload: dict = {}
                            if rating_sel != "No change":
                                try:
                                    payload["rating"] = int(rating_sel)
                                except Exception:
                                    pass
                            if feedback_text and feedback_text.strip():
                                payload["feedback_text"] = feedback_text.strip()
                            if not payload:
                                st.warning("Provide a rating or feedback text.")
                            else:
                                try:
                                    r2 = requests.patch(
                                        f"{API_BASE}/history/{record.get('id')}/feedback",
                                        headers=_auth_headers(), json=payload, timeout=10,
                                    )
                                    r2.raise_for_status()
                                    st.success("Feedback submitted")
                                    st.session_state.pop("history_loaded", None)
                                    st.rerun()
                                except requests.exceptions.HTTPError as e:
                                    st.error(f"Server error {e.response.status_code}: {e.response.text}")
                                except Exception as e:
                                    st.error(f"Failed: {e}")
