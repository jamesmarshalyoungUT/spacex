from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import SPACEX_TOOLS, get_latest_launch_external, get_next_launch_external


SYSTEM_PROMPT = """You are an AI hiring-test demo agent that answers SpaceX questions using tools.

Core rules:
- Never fabricate SpaceX facts. Always call tools for factual answers.
- If user request is ambiguous, ask a clarifying question instead of guessing.
- Prefer multiple tool calls when needed for completeness (for example launch -> rocket -> launchpad).
- Handle tool errors gracefully and explain what failed.
- Keep responses concise but data-grounded.

"""


def _format_trace(messages: list[Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                trace.append(
                    {
                        "type": "action",
                        "tool": call.get("name"),
                        "tool_input": call.get("args"),
                    }
                )
        elif isinstance(msg, ToolMessage):
            trace.append(
                {
                    "type": "observation",
                    "tool": msg.name,
                    "observation": str(msg.content),
                }
            )
    return trace


def _is_latest_launch_query(user_input: str) -> bool:
    text = user_input.lower()
    has_latest_intent = bool(re.search(r"\b(last|latest|most recent)\b", text))
    has_launch_intent = "launch" in text
    return has_latest_intent and has_launch_intent


def _is_next_launch_query(user_input: str) -> bool:
    text = user_input.lower()
    has_next_intent = bool(re.search(r"\b(next|upcoming)\b", text))
    has_launch_intent = "launch" in text
    return has_next_intent and has_launch_intent


def _needs_clarification(user_input: str) -> bool:
    text = user_input.lower().strip()
    if "launch" not in text:
        return False

    has_specific_scope = any(
        token in text
        for token in [
            "latest",
            "last",
            "next",
            "upcoming",
            "year",
            "202",
            "falcon",
            "starlink",
            "vandenberg",
            "where",
            "when",
            "how many",
            "successful",
            "outcome",
            "first",
            "most recent",
        ]
    )
    return not has_specific_scope


def _parse_iso_utc(date_value: str | None) -> datetime | None:
    if not date_value or not isinstance(date_value, str):
        return None
    cleaned = date_value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _friendly_utc(date_value: str | None) -> str:
    parsed = _parse_iso_utc(date_value)
    if not parsed:
        return date_value or "an unknown date"
    parsed = parsed.astimezone(timezone.utc)
    # Example: April 2, 2026 at 11:52 UTC
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year} at {parsed.strftime('%H:%M')} UTC"


def _extract_latest_launch_observation(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in trace:
        if entry.get("type") == "observation" and entry.get("tool") == "get_latest_launch":
            raw = entry.get("observation")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    return None
    return None


def _extract_next_launch_observation(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in trace:
        if entry.get("type") == "observation" and entry.get("tool") == "get_next_launch":
            raw = entry.get("observation")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    return None
    return None


def _extract_observation_by_tool(trace: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for entry in reversed(trace):
        if entry.get("type") == "observation" and entry.get("tool") == tool_name:
            raw = entry.get("observation")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def _add_determination(
    trace: list[dict[str, Any]], check: str, verdict: str, rationale: str, **extra: Any
) -> None:
    payload: dict[str, Any] = {
        "type": "determination",
        "check": check,
        "verdict": verdict,
        "rationale": rationale,
    }
    payload.update(extra)
    trace.append(payload)


def _quality_score_from_confidence(
    confidence: str, status: str, fallback_used: bool, issues_count: int
) -> int:
    # Base confidence signal.
    base_map = {
        "high": 88,
        "medium": 72,
        "low": 56,
    }
    score = base_map.get(str(confidence).lower(), 50)

    # Quality gate verdict impact.
    normalized_status = str(status).lower()
    if normalized_status == "pass":
        score += 8
    elif normalized_status == "fail":
        score -= 18
    else:
        score -= 5

    # Strong penalty when fallback text had to be used.
    if fallback_used:
        score -= 20

    # Penalize reported issues with a cap to avoid over-penalizing noise.
    bounded_issues = max(0, min(int(issues_count), 10))
    score -= bounded_issues * 4

    return max(0, min(100, score))


def _extract_quality_gate(trace: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in reversed(trace):
        if entry.get("type") == "determination" and entry.get("check") == "final_answer_quality_gate":
            confidence = str(entry.get("confidence", "low")).lower()
            verdict = str(entry.get("verdict", "fail")).lower()
            fallback_used = bool(entry.get("fallback_used", False))
            issues_count = int(entry.get("issues_count", 0))
            score = _quality_score_from_confidence(
                confidence=confidence,
                status=verdict,
                fallback_used=fallback_used,
                issues_count=issues_count,
            )
            return {
                "status": verdict,
                "confidence": confidence,
                "confidence_score": score,
                "fallback_used": fallback_used,
                "issues_count": issues_count,
                "summary": (
                    f"Quality Gate: {verdict.upper()} "
                    f"({confidence} confidence, score={score})"
                ),
            }

    return {
        "status": "unknown",
        "confidence": "low",
        "confidence_score": 50,
        "fallback_used": False,
        "summary": "Quality Gate: UNKNOWN (low confidence, score=50)",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


class SpaceXAgentSession:
    def __init__(self, model_name: str, temperature: float = 0, verbose: bool = True) -> None:
        self._verbose = verbose
        self._messages: list[Any] = []
        self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self._graph = create_react_agent(model=self._llm, tools=SPACEX_TOOLS, prompt=SYSTEM_PROMPT)

    def ask(self, user_input: str) -> dict[str, Any]:
        if _needs_clarification(user_input):
            trace: list[dict[str, Any]] = [
                {
                    "type": "action",
                    "tool": "clarification_guard",
                    "tool_input": {
                        "reason": "launch query lacks enough scope for a reliable answer",
                        "question": user_input,
                    },
                }
            ]
            _add_determination(
                trace,
                check="clarification_required",
                verdict="pass",
                rationale="The request is ambiguous and requires user clarification before querying tools.",
            )

            clarify_text = (
                "Happy to help. Do you mean the latest launch, the next launch, or launches in a specific year? "
                "You can also name a mission like Starlink 9-1."
            )
            quality_gate = {
                "status": "pass",
                "confidence": "high",
                "confidence_score": 100,
                "fallback_used": False,
                "issues_count": 0,
                "summary": "Quality Gate: PASS (high confidence, score=100)",
            }
            return {
                "output": clarify_text,
                "user_answer": clarify_text,
                "trace": trace,
                "quality_gate": quality_gate,
            }

        start_idx = len(self._messages)
        self._messages.append(HumanMessage(content=user_input))

        result = self._graph.invoke({"messages": self._messages})
        updated_messages = result.get("messages", [])
        new_messages = updated_messages[start_idx:]
        self._messages = updated_messages

        answer = ""
        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break
        if not answer:
            for msg in reversed(self._messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = str(msg.content)
                    break

        trace = _format_trace(new_messages)

        if _is_latest_launch_query(user_input):
            answer, trace = self._validate_latest_launch_answer(answer=answer, trace=trace)
        if _is_next_launch_query(user_input):
            answer, trace = self._validate_next_launch_answer(answer=answer, trace=trace)

        answer, trace = self._evaluate_final_answer(user_input=user_input, answer=answer, trace=trace)

        if self._verbose:
            for entry in trace:
                if entry["type"] == "action":
                    print(f"Action: {entry['tool']} | Input: {entry['tool_input']}")
                if entry["type"] == "observation":
                    print(f"Observation ({entry['tool']}): {entry['observation']}")

        quality_gate = _extract_quality_gate(trace)
        user_answer = self._build_user_friendly_answer(
            user_input=user_input,
            technical_answer=answer,
            trace=trace,
            quality_gate=quality_gate,
        )
        return {
            "output": answer,
            "user_answer": user_answer,
            "trace": trace,
            "quality_gate": quality_gate,
        }

    def _build_user_friendly_answer(
        self,
        user_input: str,
        technical_answer: str,
        trace: list[dict[str, Any]],
        quality_gate: dict[str, Any],
    ) -> str:
        next_external = _extract_observation_by_tool(trace, "get_next_launch_external")
        latest_external = _extract_observation_by_tool(trace, "get_latest_launch_external")

        if _is_next_launch_query(user_input) and next_external and not next_external.get("error"):
            name = next_external.get("name", "the next SpaceX mission")
            date_utc = _friendly_utc(next_external.get("date_utc"))
            status = next_external.get("status", "status not available")
            location = next_external.get("location")
            if location:
                return (
                    f"The next SpaceX launch is {name}, currently scheduled for {date_utc}. "
                    f"Current mission status is {status}, and it is planned from {location}."
                )
            return (
                f"The next SpaceX launch is {name}, currently scheduled for {date_utc}. "
                f"Current mission status is {status}."
            )

        if _is_latest_launch_query(user_input) and latest_external and not latest_external.get("error"):
            name = latest_external.get("name", "the latest SpaceX mission")
            date_utc = _friendly_utc(latest_external.get("date_utc"))
            status = latest_external.get("status", "status not available")
            return f"The most recent SpaceX launch was {name} on {date_utc}. Mission status: {status}."

        if quality_gate.get("status") == "fail" and quality_gate.get("fallback_used"):
            return (
                "I want to make sure you get a reliable answer. Right now I cannot confirm this with high confidence. "
                "If you want, I can run another check and try again."
            )

        prompt = (
            "Rewrite the following answer for a general audience. "
            "Use plain, friendly language and avoid technical or coding terminology. "
            "Keep it concise and factual.\n\n"
            f"User question: {user_input}\n"
            f"Answer: {technical_answer}"
        )
        rewritten = self._llm.invoke(prompt)
        rewritten_text = str(getattr(rewritten, "content", rewritten)).strip()
        return rewritten_text or technical_answer

    def _validate_latest_launch_answer(self, answer: str, trace: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        latest_launch = _extract_latest_launch_observation(trace)
        if not latest_launch:
            _add_determination(
                trace,
                check="latest_launch_trace_presence",
                verdict="skipped",
                rationale="No get_latest_launch observation found in this turn.",
            )
            return answer, trace

        primary_date = _parse_iso_utc(latest_launch.get("date_utc"))
        if not primary_date:
            _add_determination(
                trace,
                check="latest_launch_date_parse",
                verdict="failed",
                rationale="Primary latest-launch date could not be parsed.",
            )
            return answer, trace

        age_days = (datetime.now(timezone.utc) - primary_date).days
        if age_days <= 180:
            _add_determination(
                trace,
                check="latest_launch_freshness",
                verdict="pass",
                rationale=f"Primary latest-launch date is {age_days} days old (<= 180).",
            )
            return answer, trace

        _add_determination(
            trace,
            check="latest_launch_freshness",
            verdict="fail",
            rationale=f"Primary latest-launch date is {age_days} days old (> 180).",
        )

        trace.append(
            {
                "type": "action",
                "tool": "freshness_guard",
                "tool_input": {"reason": "latest launch result is older than 180 days", "age_days": age_days},
            }
        )

        external_raw = get_latest_launch_external.invoke({})
        trace.append(
            {
                "type": "observation",
                "tool": "get_latest_launch_external",
                "observation": str(external_raw),
            }
        )

        external_data: dict[str, Any] | None = None
        if isinstance(external_raw, str):
            try:
                parsed = json.loads(external_raw)
                if isinstance(parsed, dict):
                    external_data = parsed
            except json.JSONDecodeError:
                external_data = None

        if not external_data or external_data.get("error"):
            _add_determination(
                trace,
                check="latest_launch_external_cross_check",
                verdict="failed",
                rationale="Secondary source did not return a usable record.",
            )
            safe_answer = (
                "I cannot confidently confirm the latest launch from SpaceX v5 data because it appears stale "
                f"({latest_launch.get('name')} on {latest_launch.get('date_utc')}). "
                "I attempted a secondary live-source validation but it failed. "
                "Please retry, or allow a different source so I can provide a grounded latest-launch answer."
            )
            return safe_answer, trace

        external_date = _parse_iso_utc(external_data.get("date_utc"))
        if not external_date:
            _add_determination(
                trace,
                check="latest_launch_external_date_parse",
                verdict="failed",
                rationale="Secondary latest-launch date could not be parsed.",
            )
            safe_answer = (
                "SpaceX v5 latest-launch data appears stale and external validation returned an unusable date. "
                "I am not returning a guessed value."
            )
            return safe_answer, trace

        if external_date > primary_date:
            _add_determination(
                trace,
                check="latest_launch_external_newer",
                verdict="pass",
                rationale="Secondary source returned a newer launch date than primary.",
            )
            corrected_answer = (
                "I validated the result and detected stale primary data. "
                f"SpaceX v5 reported {latest_launch.get('name')} on {latest_launch.get('date_utc')}, "
                f"which is {age_days} days old. "
                "Cross-checking a secondary live source (Launch Library 2) shows a newer SpaceX launch: "
                f"{external_data.get('name')} on {external_data.get('date_utc')} "
                f"(status: {external_data.get('status')}). "
                "I am using the newer validated value instead of the stale one."
            )
            return corrected_answer, trace

        _add_determination(
            trace,
            check="latest_launch_external_newer",
            verdict="failed",
            rationale="Secondary source did not provide a newer launch than primary.",
        )

        conservative_answer = (
            "I ran freshness validation because the primary SpaceX v5 result looked old. "
            "The secondary source did not show a newer launch than the primary result, so I cannot assert a newer value "
            "without evidence."
        )
        return conservative_answer, trace

    def _validate_next_launch_answer(self, answer: str, trace: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        next_launch = _extract_next_launch_observation(trace)
        if not next_launch:
            _add_determination(
                trace,
                check="next_launch_trace_presence",
                verdict="skipped",
                rationale="No get_next_launch observation found in this turn.",
            )
            return answer, trace

        primary_date = _parse_iso_utc(next_launch.get("date_utc"))
        if not primary_date:
            _add_determination(
                trace,
                check="next_launch_date_parse",
                verdict="failed",
                rationale="Primary next-launch date could not be parsed.",
            )
            return answer, trace

        now_utc = datetime.now(timezone.utc)
        # "Next" launch must not be in the past. If it is, primary source is stale for this query.
        if primary_date >= now_utc:
            _add_determination(
                trace,
                check="next_launch_future_guard",
                verdict="pass",
                rationale="Primary next-launch date is in the future.",
            )
            return answer, trace

        _add_determination(
            trace,
            check="next_launch_future_guard",
            verdict="fail",
            rationale="Primary next-launch date is in the past.",
        )

        age_days = (now_utc - primary_date).days
        trace.append(
            {
                "type": "action",
                "tool": "future_guard",
                "tool_input": {
                    "reason": "next launch returned with past date",
                    "age_days": age_days,
                },
            }
        )

        external_raw = get_next_launch_external.invoke({})
        trace.append(
            {
                "type": "observation",
                "tool": "get_next_launch_external",
                "observation": str(external_raw),
            }
        )

        external_data: dict[str, Any] | None = None
        if isinstance(external_raw, str):
            try:
                parsed = json.loads(external_raw)
                if isinstance(parsed, dict):
                    external_data = parsed
            except json.JSONDecodeError:
                external_data = None

        if not external_data or external_data.get("error"):
            _add_determination(
                trace,
                check="next_launch_external_cross_check",
                verdict="failed",
                rationale="Secondary source did not return a usable upcoming launch.",
            )
            safe_answer = (
                "I cannot confidently confirm the next launch from SpaceX v5 because it returned a past date "
                f"({next_launch.get('name')} on {next_launch.get('date_utc')}). "
                "I attempted a secondary live-source upcoming-launch validation but it failed. "
                "I am avoiding a guessed answer."
            )
            return safe_answer, trace

        external_date = _parse_iso_utc(external_data.get("date_utc"))
        if not external_date or external_date < now_utc:
            _add_determination(
                trace,
                check="next_launch_external_future_guard",
                verdict="failed",
                rationale="Secondary source did not return a future launch date.",
            )
            safe_answer = (
                "The primary source returned a past date for next launch, and secondary validation did not provide "
                "a reliable future launch date. I am not returning an ungrounded value."
            )
            return safe_answer, trace

        _add_determination(
            trace,
            check="next_launch_external_future_guard",
            verdict="pass",
            rationale="Secondary source returned a future upcoming launch date.",
        )

        corrected_answer = (
            "I validated the result and detected stale primary next-launch data. "
            f"SpaceX v5 reported {next_launch.get('name')} on {next_launch.get('date_utc')} (past date). "
            "Cross-checking a secondary live source (Launch Library 2) shows the next upcoming SpaceX launch as "
            f"{external_data.get('name')} on {external_data.get('date_utc')} "
            f"(status: {external_data.get('status')}). "
            "I am using the validated upcoming value instead of the stale one."
        )
        return corrected_answer, trace

    def _evaluate_final_answer(
        self, user_input: str, answer: str, trace: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        evaluator_input = {
            "user_question": user_input,
            "candidate_answer": answer,
            "trace": trace,
        }

        trace.append(
            {
                "type": "action",
                "tool": "final_answer_evaluator_agent",
                "tool_input": {
                    "objective": "Validate final answer for intent match and grounding in observed tool data",
                    "question": user_input,
                },
            }
        )

        evaluator_prompt = (
            "You are a strict QA evaluator for an agent answer.\n"
            "Evaluate whether the candidate answer matches user intent and is grounded in the provided trace.\n"
            "Do not invent facts.\n"
            "Return JSON only with keys: intent_match, grounded_in_trace, verdict, confidence, issues, recommended_action, revised_answer.\n"
            "- intent_match: true/false\n"
            "- grounded_in_trace: true/false\n"
            "- verdict: pass/fail\n"
            "- confidence: low/medium/high\n"
            "- issues: array of short strings\n"
            "- recommended_action: keep|revise|ask_clarification\n"
            "- revised_answer: string (empty if keep)\n\n"
            f"Input JSON:\n{json.dumps(evaluator_input, ensure_ascii=True)}"
        )

        eval_raw = self._llm.invoke(evaluator_prompt)
        eval_text = str(getattr(eval_raw, "content", eval_raw))
        eval_data = _extract_json_object(eval_text)

        trace.append(
            {
                "type": "observation",
                "tool": "final_answer_evaluator_agent",
                "observation": eval_text,
            }
        )

        if not eval_data:
            _add_determination(
                trace,
                check="final_answer_evaluator_parse",
                verdict="failed",
                rationale="Evaluator output was not valid JSON; keeping existing answer.",
            )
            return answer, trace

        intent_match = bool(eval_data.get("intent_match"))
        grounded_in_trace = bool(eval_data.get("grounded_in_trace"))
        verdict = str(eval_data.get("verdict", "fail")).lower()
        recommended_action = str(eval_data.get("recommended_action", "keep")).lower()
        confidence = str(eval_data.get("confidence", "low"))
        issues = eval_data.get("issues", [])

        if intent_match and grounded_in_trace and verdict == "pass":
            _add_determination(
                trace,
                check="final_answer_quality_gate",
                verdict="pass",
                rationale=(
                    f"Evaluator passed answer (confidence={confidence}, issues={issues})."
                ),
                confidence=confidence,
                fallback_used=False,
                issues_count=len(issues) if isinstance(issues, list) else 0,
            )
            return answer, trace

        revised_answer = str(eval_data.get("revised_answer", "") or "").strip()
        if recommended_action == "revise" and revised_answer:
            _add_determination(
                trace,
                check="final_answer_quality_gate",
                verdict="fail",
                rationale=(
                    f"Evaluator requested revision (confidence={confidence}, issues={issues}). "
                    "Using evaluator-proposed revised answer."
                ),
                confidence=confidence,
                fallback_used=False,
                issues_count=len(issues) if isinstance(issues, list) else 0,
            )
            return revised_answer, trace

        fallback = (
            "I cannot confidently provide a final answer that is fully validated against current tool evidence. "
            "Please allow me to run additional checks or clarify your request so I can return a grounded answer."
        )
        _add_determination(
            trace,
            check="final_answer_quality_gate",
            verdict="fail",
            rationale=(
                f"Evaluator failed answer (confidence={confidence}, issues={issues}, "
                f"recommended_action={recommended_action}). Using safe fallback."
            ),
            confidence=confidence,
            fallback_used=True,
            issues_count=len(issues) if isinstance(issues, list) else 0,
        )
        return fallback, trace


def build_agent_session(verbose: bool = True) -> SpaceXAgentSession:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in .env before running the agent.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return SpaceXAgentSession(model_name=model_name, temperature=0, verbose=verbose)
