from __future__ import annotations

import json
import os

import streamlit as st

from src.agent import build_agent_session


SAMPLE_QUESTIONS = [
    "When was the last SpaceX launch?",
    "What's the next SpaceX launch and where is it happening?",
    "How many launches did SpaceX complete in 2024?",
    "Which rocket was used for the Starlink 9-1 mission?",
    "Show me all successful Falcon 9 launches.",
    "What was the outcome of the first Falcon Heavy launch?",
    "Tell me about the most recent launch from Vandenberg.",
]


def _apply_streamlit_secrets() -> None:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    gemini_model = st.secrets.get("GEMINI_MODEL")
    spacex_api_base_url = st.secrets.get("SPACEX_API_BASE_URL")

    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = str(gemini_api_key)
        # Keep compatibility with SDKs that expect GOOGLE_API_KEY.
        os.environ["GOOGLE_API_KEY"] = str(gemini_api_key)
    if gemini_model:
        os.environ["GEMINI_MODEL"] = str(gemini_model)
    if spacex_api_base_url:
        os.environ["SPACEX_API_BASE_URL"] = str(spacex_api_base_url)

    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "Missing GEMINI_API_KEY. Add it to .streamlit/secrets.toml for local use "
            "or in Streamlit Cloud app settings under Secrets."
        )
        st.stop()


def _init_state() -> None:
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = build_agent_session(verbose=False)
    if "history" not in st.session_state:
        st.session_state.history = []


def _render_trace(trace: list[dict]) -> None:
    if not trace:
        st.caption("No trace steps returned for this turn.")
        return

    step = 0
    for item in trace:
        if item.get("type") == "action":
            step += 1
            with st.expander(f"Step {step}: Action - {item.get('tool')}", expanded=True):
                st.code(json.dumps(item.get("tool_input"), indent=2, default=str), language="json")
        elif item.get("type") == "observation":
            with st.expander(f"Step {step}: Observation - {item.get('tool')}", expanded=False):
                st.text(item.get("observation", ""))


def main() -> None:
    st.set_page_config(page_title="SpaceX Agentic Streamlit Demo", layout="wide")
    st.title("SpaceX Agentic Streamlit Demo")
    st.caption("Conversational SpaceX assistant with visible Action/Observation trace.")

    _apply_streamlit_secrets()
    _init_state()

    with st.sidebar:
        st.subheader("Quick Prompts")
        selected = st.selectbox("Interview sample question", options=["Select one"] + SAMPLE_QUESTIONS)
        if st.button("Start New Session"):
            st.session_state.agent_session = build_agent_session(verbose=False)
            st.session_state.history = []
            st.rerun()

    user_input = st.chat_input("Ask a SpaceX question...")

    if selected != "Select one" and not user_input:
        user_input = selected

    if user_input:
        with st.spinner("Agent is reasoning and calling tools..."):
            result = st.session_state.agent_session.ask(user_input)

        st.session_state.history.append(
            {
                "question": user_input,
                "answer": result.get("output", ""),
                "trace": result.get("trace", []),
            }
        )

    if not st.session_state.history:
        st.info("Ask a question or pick a sample prompt to begin.")
        return

    for idx, turn in enumerate(st.session_state.history, start=1):
        with st.container(border=True):
            st.markdown(f"**Turn {idx}**")
            st.markdown(f"**User:** {turn['question']}")
            st.markdown(f"**Agent:** {turn['answer']}")
            st.markdown("**Trace Timeline**")
            _render_trace(turn.get("trace", []))


if __name__ == "__main__":
    main()
