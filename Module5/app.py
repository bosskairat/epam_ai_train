import streamlit as st
from agents.orchestrator import query_agent

st.title("Weather and News Query App")

if "history" not in st.session_state:
    st.session_state.history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False


# ✅ clear BEFORE widget creation
if st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.clear_input = False


with st.form("query_form"):
    col1, col2 = st.columns([6, 1])

    with col1:
        st.text_input(
            "Enter your question",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Enter your question..."
        )
    with col2:
        submit = st.form_submit_button("Ask")


if submit and st.session_state.user_input:
    response = query_agent(st.session_state.user_input)
    st.session_state.history.append(
        (st.session_state.user_input, response)
    )

    # mark for clearing on next run
    st.session_state.clear_input = True
    st.rerun()


for q, r in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {r}")
    st.divider()
