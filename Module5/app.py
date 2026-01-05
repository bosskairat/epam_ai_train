import streamlit as st
from agents.context import ConversationContext
from agents.orchestrator import AgentOrchestrator

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Weather & News Agent",
    layout="centered"
)

st.title("🌍 Weather & News AI Agent")

# -------------------------------
# Session state initialization
# -------------------------------
if "context" not in st.session_state:
    st.session_state.context = ConversationContext()

if "agent" not in st.session_state:
    st.session_state.agent = AgentOrchestrator()

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# -------------------------------
# Clear input BEFORE widget creation
# -------------------------------
if st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.clear_input = False

# ===============================
# 🟢 CHAT HISTORY (TOP)
# ===============================
chat_container = st.container()

with chat_container:
    if not st.session_state.context.history:
        st.info("Ask a question about weather or news 👇")

    for msg in st.session_state.context.history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:**\n\n{msg['content']}")
            st.divider()

# ===============================
# 🟢 INPUT FORM (BOTTOM)
# ===============================
with st.form("query_form", clear_on_submit=False):
    col1, col2 = st.columns([6, 1])

    with col1:
        st.text_input(
            label="Ask",
            key="user_input",
            placeholder="Ask about weather or news...",
            label_visibility="collapsed"
        )

    with col2:
        submit = st.form_submit_button("Ask")

# -------------------------------
# Handle submission
# -------------------------------
if submit and st.session_state.user_input.strip():
    query = st.session_state.user_input.strip()

    st.session_state.context.add("user", query)

    response = st.session_state.agent.handle(
        query,
        st.session_state.context.history
    )

    st.session_state.context.add("assistant", response)

    st.session_state.clear_input = True
    st.rerun()
