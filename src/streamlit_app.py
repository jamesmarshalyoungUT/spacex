from __future__ import annotations

import json
import os
from datetime import datetime, timezone

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


def _friendly_utc(date_value: str) -> str | None:
    try:
        parsed = datetime.fromisoformat(date_value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year} at {parsed.strftime('%H:%M')} UTC"


def _collect_friendly_dates(obj: object, prefix: str = "") -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str) and ("date_utc" in key or key in {"net", "date"}):
                friendly = _friendly_utc(value)
                if friendly:
                    results.append((path, value, friendly))
            results.extend(_collect_friendly_dates(value, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            results.extend(_collect_friendly_dates(item, path))

    return results


def _friendly_date_hints(observation: str) -> list[tuple[str, str, str]]:
    try:
        parsed = json.loads(observation)
    except json.JSONDecodeError:
        return []
    return _collect_friendly_dates(parsed)


def _flow_segment(tool: str | None = None, check: str | None = None) -> str:
    tool_name = str(tool or "")
    check_name = str(check or "")

    if tool_name in {
        "get_latest_launch",
        "get_next_launch",
        "search_launches_by_name",
        "get_launches_in_year",
        "get_successful_launches_by_rocket",
        "get_rocket_by_id",
        "get_launchpad_by_id",
        "get_recent_launches_from_location",
    }:
        return "C-D-E"

    if tool_name in {"freshness_guard", "future_guard"}:
        return "F-G"

    if tool_name in {"get_latest_launch_external", "get_next_launch_external"}:
        return "G-L-M/O"

    if tool_name in {"website_lookup_consent_prompt", "website_lookup_user_response"}:
        return "G-H-I"

    if tool_name in {"spacex_website_latest_lookup", "spacex_website_next_lookup", "spacex_website_lookup"}:
        return "I-J"

    if tool_name == "final_answer_evaluator_agent":
        return "P-Q"

    if check_name.startswith("next_launch") or check_name.startswith("latest_launch"):
        return "F-G"
    if check_name.startswith("website_lookup_consent"):
        return "G-H-I"
    if check_name.startswith("spacex_website_lookup"):
        return "I-J"
    if check_name.startswith("final_answer"):
        return "P-Q"

    return "A-B"


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
            segment = _flow_segment(tool=str(item.get("tool")))
            with st.expander(f"Step {step} ({segment}): Action - {item.get('tool')}", expanded=True):
                st.code(json.dumps(item.get("tool_input"), indent=2, default=str), language="json")
        elif item.get("type") == "observation":
            segment = _flow_segment(tool=str(item.get("tool")))
            with st.expander(f"Step {step} ({segment}): Observation - {item.get('tool')}", expanded=False):
                observation_text = str(item.get("observation", ""))
                st.text(observation_text)
                hints = _friendly_date_hints(observation_text)
                if hints:
                    st.markdown("**Friendly Dates**")
                    for path, raw_value, friendly in hints:
                        st.markdown(f"- `{path}`: `{raw_value}` -> `{friendly}`")
        elif item.get("type") == "determination":
            check = item.get("check", "unknown_check")
            verdict = str(item.get("verdict", "unknown")).upper()
            rationale = item.get("rationale", "")
            segment = _flow_segment(check=str(check))
            with st.expander(f"Step {step} ({segment}): Determination - {check}", expanded=True):
                st.markdown(f"**Verdict:** {verdict}")
                st.markdown(f"**Rationale:** {rationale}")


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
                "answer": result.get("user_answer", result.get("output", "")),
                "trace": result.get("trace", []),
                "quality_gate": result.get("quality_gate", {}),
                "qa_review": result.get("qa_review", {}),
            }
        )

    if not st.session_state.history:
        st.info("Ask a question or pick a sample prompt to begin.")
        return

    for idx, turn in enumerate(st.session_state.history, start=1):
        with st.container(border=True):
            st.markdown(f"**Turn {idx}**")
            st.markdown(f"**User:** {turn['question']}")
            qg = turn.get("quality_gate", {})
            qg_status = str(qg.get("status", "unknown")).upper()
            qg_conf = str(qg.get("confidence", "low")).lower()
            qg_score = qg.get("confidence_score", 50)
            qg_fallback = bool(qg.get("fallback_used", False))
            if qg_fallback:
                st.markdown(
                    f"**Quality Gate:** {qg_status} ({qg_conf} confidence, score={qg_score}, fallback used)"
                )
            else:
                st.markdown(f"**Quality Gate:** {qg_status} ({qg_conf} confidence, score={qg_score})")
            qa_review = turn.get("qa_review", {})
            if qa_review:
                st.markdown("**QA Evaluator Review**")
                st.code(json.dumps(qa_review, indent=2, default=str), language="json")
            st.markdown(f"**Agent:** {turn['answer']}")
            st.markdown("**Trace Timeline**")
            _render_trace(turn.get("trace", []))


if __name__ == "__main__":
    main()
