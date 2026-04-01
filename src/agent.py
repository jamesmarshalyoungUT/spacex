from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import (
    SPACEX_TOOLS,
    get_latest_launch_external,
    get_next_launch_external,
    _latest_spacex_launch_from_spacex_website,
    _next_spacex_launch_from_spacex_website,
)


SYSTEM_PROMPT = """You are an AI agent that answers SpaceX questions using tools.

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


def _is_failed_launch_query(user_input: str) -> bool:
    text = user_input.lower()
    return bool(re.search(r"\b(fail|failed|failure|unsuccessful)\b", text)) or _has_fuzzy_keyword(text, ["failed", "failure", "unsuccessful"])


def _is_latest_launch_query(user_input: str) -> bool:
    if _is_failed_launch_query(user_input):
        return False
    text = user_input.lower()
    has_latest_intent = bool(re.search(r"\b(last|latest)\b", text)) or _has_fuzzy_keyword(text, ["last", "latest", "recent"])
    has_launch_intent = _has_launch_intent(text)
    return has_latest_intent and has_launch_intent


def _is_next_launch_query(user_input: str) -> bool:
    text = user_input.lower()
    has_next_intent = bool(re.search(r"\b(next|upcoming)\b", text)) or _has_fuzzy_keyword(text, ["next", "upcoming"])
    has_launch_intent = _has_launch_intent(text)
    return has_next_intent and has_launch_intent


def _is_year_based_query(user_input: str) -> bool:
    if not _has_launch_intent(user_input):
        return False
    return _extract_year_from_text(user_input) is not None


def _is_mission_specific_query(user_input: str) -> bool:
    if not _has_launch_intent(user_input):
        return False
    # Exclude queries that are already handled by other validators
    if _is_latest_launch_query(user_input):
        return False
    if _is_next_launch_query(user_input):
        return False
    if _is_year_based_query(user_input):
        return False
    if _is_failed_launch_query(user_input):
        return False
    # Mission-specific queries have specific mission/payload names or ask about specific details
    text = user_input.lower()
    mission_keywords = ["starlink", "falcon", "rocket", "satellite", "payload", "mission", "used", "rocket"]
    return any(keyword in text for keyword in mission_keywords)


def _extract_year_from_text(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _needs_clarification(user_input: str) -> bool:
    text = user_input.lower().strip()
    if not _has_launch_intent(text):
        return False

    year = _extract_year_from_text(text)
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
    if year is not None:
        has_specific_scope = True
    return not has_specific_scope


def _has_launch_intent(text: str) -> bool:
    lowered = text.lower()
    if "launch" in lowered:
        return True
    return _has_fuzzy_keyword(lowered, ["launch"])


def _has_fuzzy_keyword(text: str, keywords: list[str], threshold: float = 0.76) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    for token in tokens:
        for keyword in keywords:
            if token == keyword:
                return True
            # Quick length gate reduces false positives and unnecessary comparisons.
            if abs(len(token) - len(keyword)) > 2:
                continue
            if SequenceMatcher(None, token, keyword).ratio() >= threshold:
                return True
    return False


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


def _extract_list_observation_by_tool(trace: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]] | None:
    for entry in reversed(trace):
        if entry.get("type") == "observation" and entry.get("tool") == tool_name:
            raw = entry.get("observation")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [item for item in parsed if isinstance(item, dict)]
                except json.JSONDecodeError:
                    return None
    return None


def _build_supported_engagement_followup(user_input: str, trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    failed_launch = _extract_observation_by_tool(trace, "get_last_failed_launch")
    next_external = _extract_observation_by_tool(trace, "get_next_launch_external")
    next_primary = _extract_observation_by_tool(trace, "get_next_launch")
    latest_external = _extract_observation_by_tool(trace, "get_latest_launch_external")
    latest_primary = _extract_observation_by_tool(trace, "get_latest_launch")
    website_lookup = _extract_observation_by_tool(trace, "spacex_website_lookup")
    requested_year = _extract_year_from_text(user_input)

    def _year_summary_followup(launch_data: dict[str, Any], rationale: str) -> dict[str, Any] | None:
        date_utc = launch_data.get("date_utc")
        parsed = _parse_iso_utc(date_utc) if isinstance(date_utc, str) else None
        year = parsed.year if parsed else None
        if year is None or year == requested_year:
            return None
        return {
            "include_followup": True,
            "offer_text": f"I can summarize other SpaceX launches from {year}",
            "suggested_query": f"What SpaceX launches happened in {year}?",
            "rationale": rationale,
        }

    if failed_launch and not failed_launch.get("error") and failed_launch.get("name"):
        year_followup = _year_summary_followup(
            failed_launch,
            "Derived a supported follow-up from the failed launch result using the launches-in-year tool.",
        )
        if year_followup:
            return year_followup
        return {
            "include_followup": True,
            "offer_text": "I can tell you about the next scheduled SpaceX launch",
            "suggested_query": "When is the next SpaceX launch?",
            "rationale": "Derived a supported follow-up from the failed launch result using the next-launch tool.",
        }

    next_launch = next_external or next_primary
    if next_launch and not next_launch.get("error") and next_launch.get("name"):
        year_followup = _year_summary_followup(
            next_launch,
            "Derived a supported follow-up from the next-launch result using the launches-in-year tool.",
        )
        if year_followup:
            return year_followup
        return {
            "include_followup": True,
            "offer_text": "I can tell you about the most recent completed SpaceX launch",
            "suggested_query": "When was the latest SpaceX launch?",
            "rationale": "Derived a supported follow-up from the next-launch result using the latest-launch tool.",
        }

    latest_launch = website_lookup or latest_external or latest_primary
    if latest_launch and not latest_launch.get("error") and latest_launch.get("name"):
        year_followup = _year_summary_followup(
            latest_launch,
            "Derived a supported follow-up from the latest-launch result using the launches-in-year tool.",
        )
        if year_followup:
            return year_followup
        return {
            "include_followup": True,
            "offer_text": "I can tell you about the most recent SpaceX launch failure",
            "suggested_query": "When was the last failed SpaceX launch?",
            "rationale": "Derived a supported follow-up from the latest-launch result using the failed-launch tool.",
        }

    return None


def _has_failed_check(trace: list[dict[str, Any]], check: str) -> bool:
    for entry in trace:
        if (
            entry.get("type") == "determination"
            and entry.get("check") == check
            and str(entry.get("verdict", "")).lower() in {"fail", "failed"}
        ):
            return True
    return False


def _safe_source_line(source_name: str, data: dict[str, Any] | None) -> str:
    if not data:
        return f"- {source_name}: no result returned"

    if data.get("error"):
        details = data.get("details")
        if details:
            return f"- {source_name}: error ({data.get('error')}; details: {details})"
        return f"- {source_name}: error ({data.get('error')})"

    name = data.get("name", "unknown mission")
    raw_date = data.get("date_utc")
    date_friendly = _friendly_utc(raw_date)
    return f"- {source_name}: {name} on {date_friendly}"


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


def _extract_qa_review(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(trace):
        if entry.get("type") == "observation" and entry.get("tool") == "final_answer_evaluator_agent":
            observation = str(entry.get("observation", ""))
            return _extract_json_object(observation)
    return None


def _source_label(source: str | None) -> str:
    labels = {
        "launch_library_2": "Launch Library 2",
        "rocketlaunch_live": "RocketLaunch.Live",
        "spacex_website": "SpaceX website",
    }
    return labels.get(str(source), str(source) if source else "secondary live source")


def _has_user_prompt_question(text: str) -> bool:
    lowered = text.lower()
    return "would you like me to" in lowered or "reply yes or no" in lowered


def _is_affirmative(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(re.search(r"\b(yes|yep|yeah|sure|ok|okay|please do|go ahead|search there)\b", normalized))


def _is_negative(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(re.search(r"\b(no|nope|not now|don't|do not|stop|cancel)\b", normalized))


class SpaceXAgentSession:
    def __init__(self, model_name: str, temperature: float = 0, verbose: bool = True) -> None:
        self._verbose = verbose
        self._messages: list[Any] = []
        self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self._graph = create_react_agent(model=self._llm, tools=SPACEX_TOOLS, prompt=SYSTEM_PROMPT)
        self._pending_website_lookup: dict[str, Any] | None = None
        self._pending_engagement_action: dict[str, Any] | None = None

    def _invoke_with_timeout(self, fn: Any, timeout_seconds: int = 12) -> dict[str, Any]:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn)
                result = future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            return {
                "source": "spacex_website",
                "error": "SpaceX website lookup timed out",
                "details": f"Request exceeded {timeout_seconds} seconds",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "source": "spacex_website",
                "error": "SpaceX website lookup failed",
                "details": str(exc),
            }

        if isinstance(result, dict):
            return result
        return {
            "source": "spacex_website",
            "error": "SpaceX website lookup returned invalid payload",
            "details": str(type(result)),
        }

    def _finalize_confirmation_turn(
        self, user_input: str, answer: str, trace: list[dict[str, Any]], engagement_context: str = ""
    ) -> dict[str, Any]:
        _evaluated_answer, evaluated_trace = self._evaluate_final_answer(
            user_input=user_input,
            answer=answer,
            trace=trace,
            allow_answer_override=False,
        )
        quality_gate = _extract_quality_gate(evaluated_trace)
        qa_review = _extract_qa_review(evaluated_trace) or {
            "verdict": "unknown",
            "reason": "evaluator output not available",
        }
        user_answer = self._maybe_add_engagement_followup(
            user_input=engagement_context or user_input,
            user_answer=answer,
            trace=evaluated_trace,
        )
        return {
            "output": answer,
            "user_answer": user_answer,
            "trace": evaluated_trace,
            "quality_gate": quality_gate,
            "qa_review": qa_review,
        }

    def _maybe_add_engagement_followup(
        self,
        user_input: str,
        user_answer: str,
        trace: list[dict[str, Any]],
    ) -> str:
        # Skip if the answer itself is already a question (e.g. consent prompt).
        if _has_user_prompt_question(user_answer):
            return user_answer

        supported_followup = _build_supported_engagement_followup(user_input, trace)
        trace.append(
            {
                "type": "action",
                "tool": "engagement_followup_agent",
                "tool_input": {
                    "objective": "Suggest one short optional next SpaceX action to keep conversation going.",
                    "user_question": user_input,
                },
            }
        )

        if supported_followup:
            raw_text = json.dumps(supported_followup, ensure_ascii=True)
            trace.append(
                {
                    "type": "observation",
                    "tool": "engagement_followup_agent",
                    "observation": raw_text,
                }
            )
            payload = supported_followup
        else:
            prompt = (
                "You are an engagement follow-up agent for a SpaceX assistant.\n"
                "Given the user question and assistant answer, propose one next action the agent can perform to keep the conversation going.\n"
                "Return JSON only with keys: include_followup (true/false), offer_text, suggested_query, rationale.\n"
                "Rules:\n"
                "- Always set include_followup=true. Even if the current answer was incomplete or failed, suggest a related SpaceX topic.\n"
                "- offer_text: a short phrase starting with 'I can' describing what the agent will look up. Keep under 18 words.\n"
                "- suggested_query: the exact natural-language question the agent should answer if the user says yes.\n"
                "- Only suggest actions answerable with current reliable tools: latest launch, next launch, last failed launch, or launches in a year.\n"
                "- Do not suggest rocket details, launchpad details, location-based queries, investigations, causes, biographies, company history, or any topic not directly answerable with those tools.\n"
                "- Must be SpaceX-domain. Prefer topics related to the user's question.\n"
                "- Do not repeat a question that was just answered.\n\n"
                f"User question: {user_input}\n"
                f"Assistant answer: {user_answer}"
            )

            raw = self._llm.invoke(prompt)
            raw_text = str(getattr(raw, "content", raw))
            trace.append(
                {
                    "type": "observation",
                    "tool": "engagement_followup_agent",
                    "observation": raw_text,
                }
            )

            payload = _extract_json_object(raw_text)
        if not payload:
            _add_determination(
                trace,
                check="engagement_followup",
                verdict="skipped",
                rationale="Follow-up agent output was not valid JSON.",
            )
            return user_answer

        include = bool(payload.get("include_followup"))
        offer = str(payload.get("offer_text", "") or "").strip()
        suggested_query = str(payload.get("suggested_query", "") or "").strip()
        if not include or not offer or not suggested_query:
            _add_determination(
                trace,
                check="engagement_followup",
                verdict="skipped",
                rationale="No relevant follow-up action suggested for this turn.",
            )
            return user_answer

        self._pending_engagement_action = {"suggested_query": suggested_query}
        _add_determination(
            trace,
            check="engagement_followup",
            verdict="pass",
            rationale="Offered an optional follow-up action to encourage continued interaction.",
        )
        return f"{user_answer}\n\nIf you're interested, {offer}. Would you like me to do that?"

    def _continue_pending_engagement_action(self, user_input: str) -> dict[str, Any]:
        pending = self._pending_engagement_action or {}
        suggested_query = pending.get("suggested_query", "")
        self._pending_engagement_action = None
        trace: list[dict[str, Any]] = [
            {
                "type": "action",
                "tool": "engagement_followup_response",
                "tool_input": {"response": user_input},
            }
        ]

        if _is_negative(user_input):
            _add_determination(
                trace,
                check="engagement_followup_consent",
                verdict="declined",
                rationale="User declined the engagement follow-up action.",
            )
            answer = "No problem! Feel free to ask anything else about SpaceX."
            return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace)

        if not _is_affirmative(user_input):
            return self.ask(user_input)

        _add_determination(
            trace,
            check="engagement_followup_consent",
            verdict="granted",
            rationale="User approved engagement follow-up; executing suggested query.",
        )
        # Execute the suggested query through the normal agent flow.
        return self.ask(suggested_query)

    def _continue_pending_website_lookup(self, user_input: str) -> dict[str, Any]:
        pending = self._pending_website_lookup or {}
        query_type = pending.get("query_type")
        raw_primary_data = pending.get("primary_data")
        primary_data: dict[str, Any] = raw_primary_data if isinstance(raw_primary_data, dict) else {}
        raw_proposed_data = pending.get("proposed_data")
        proposed_data: dict[str, Any] = raw_proposed_data if isinstance(raw_proposed_data, dict) else {}
        original_query: str = str(pending.get("original_query") or "")
        trace: list[dict[str, Any]] = [
            {
                "type": "action",
                "tool": "website_lookup_user_response",
                "tool_input": {"response": user_input},
            }
        ]

        if _is_negative(user_input):
            self._pending_website_lookup = None
            if proposed_data and not proposed_data.get("error"):
                trace.append(
                    {
                        "type": "observation",
                        "tool": "secondary_validated_answer_cache",
                        "observation": json.dumps(proposed_data, ensure_ascii=True),
                    }
                )
            _add_determination(
                trace,
                check="website_lookup_consent",
                verdict="declined",
                rationale="User declined searching SpaceX website.",
            )
            if proposed_data and not proposed_data.get("error"):
                source_name = _source_label(proposed_data.get("source"))
                answer = (
                    "Understood. I will not search the SpaceX website. "
                    f"Using the previously validated non-SpaceX source ({source_name}), "
                    f"the answer is {proposed_data.get('name', 'the mission')} on "
                    f"{_friendly_utc(proposed_data.get('date_utc'))} "
                    f"(status: {proposed_data.get('status', 'status not available')})."
                )
            else:
                answer = "Understood. I will not search the SpaceX website."
            return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace, engagement_context=original_query)

        if not _is_affirmative(user_input):
            _add_determination(
                trace,
                check="website_lookup_consent",
                verdict="declined",
                rationale="User response was unclear; treating as a new question and clearing pending state.",
            )
            self._pending_website_lookup = None
            return self.ask(user_input)

        _add_determination(
            trace,
            check="website_lookup_consent",
            verdict="granted",
            rationale="User approved SpaceX website lookup.",
        )

        if query_type == "latest":
            trace.append(
                {
                    "type": "action",
                    "tool": "spacex_website_latest_lookup",
                    "tool_input": {"reason": "user approved website lookup"},
                }
            )
            website_data = self._invoke_with_timeout(_latest_spacex_launch_from_spacex_website)
        elif query_type == "next":
            trace.append(
                {
                    "type": "action",
                    "tool": "spacex_website_next_lookup",
                    "tool_input": {"reason": "user approved website lookup"},
                }
            )
            website_data = self._invoke_with_timeout(_next_spacex_launch_from_spacex_website)
        elif query_type in ("year", "mission"):
            # Year-based and mission-specific queries don't have dedicated website lookups; return early with explanation
            _add_determination(
                trace,
                check="spacex_website_lookup",
                verdict="skipped",
                rationale=f"{query_type.capitalize()}-based queries are not available via website lookup; only primary source supported.",
            )
            query_name = "year-based launch" if query_type == "year" else "mission-specific information"
            answer = (
                f"I apologize, but {query_name} lookups are only available from my primary source. "
                "Unfortunately, no data was found. "
                "You can try searching for the latest launch or next scheduled launch instead."
            )
            return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace, engagement_context=original_query)
        else:
            # Fallback for unknown query types
            _add_determination(
                trace,
                check="spacex_website_lookup",
                verdict="skipped",
                rationale="Query type not recognized for website lookup.",
            )
            answer = "I apologize, but I cannot search for this information on the SpaceX website right now."
            return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace, engagement_context=original_query)

        trace.append(
            {
                "type": "observation",
                "tool": "spacex_website_lookup",
                "observation": json.dumps(website_data, ensure_ascii=True),
            }
        )

        self._pending_website_lookup = None

        if website_data.get("error"):
            _add_determination(
                trace,
                check="spacex_website_lookup",
                verdict="failed",
                rationale=f"SpaceX website lookup failed: {website_data.get('error')}",
            )
            base = proposed_data if proposed_data else primary_data
            base_name = base.get("name", "the mission")
            base_date = _friendly_utc(base.get("date_utc"))
            base_source = _source_label(base.get("source"))
            answer = (
                "I tried checking the SpaceX website, but I could not retrieve a reliable result right now. "
                f"The best previously verified result was {base_name} on {base_date} (source: {base_source}). "
                "Please try again shortly."
            )
            return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace)

        _add_determination(
            trace,
            check="spacex_website_lookup",
            verdict="pass",
            rationale="SpaceX website lookup returned a usable record.",
        )

        name = website_data.get("name", "the mission")
        date_utc = _friendly_utc(website_data.get("date_utc"))
        status = website_data.get("status", "status not available")

        if query_type == "latest":
            answer = f"Thanks for confirming. From the SpaceX website, the latest launch is {name} on {date_utc}. Status: {status}."
        else:
            answer = f"Thanks for confirming. From the SpaceX website, the next launch is {name}, scheduled for {date_utc}. Status: {status}."

        return self._finalize_confirmation_turn(user_input=user_input, answer=answer, trace=trace)

    def ask(self, user_input: str) -> dict[str, Any]:
        if self._pending_website_lookup:
            return self._continue_pending_website_lookup(user_input)

        if self._pending_engagement_action:
            return self._continue_pending_engagement_action(user_input)

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
                "qa_review": {
                    "verdict": "skipped",
                    "reason": "clarification requested before evaluation",
                },
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
            answer, trace = self._validate_latest_launch_answer(answer=answer, trace=trace, user_input=user_input)
        if _is_next_launch_query(user_input):
            answer, trace = self._validate_next_launch_answer(answer=answer, trace=trace, user_input=user_input)
        if _is_year_based_query(user_input):
            answer, trace = self._validate_year_based_answer(answer=answer, trace=trace, user_input=user_input)
        if _is_mission_specific_query(user_input):
            answer, trace = self._validate_mission_specific_answer(answer=answer, trace=trace, user_input=user_input)

        if self._pending_website_lookup is not None:
            quality_gate = {
                "status": "pass",
                "confidence": "high",
                "confidence_score": 100,
                "fallback_used": False,
                "issues_count": 0,
                "summary": "Quality Gate: PASS (high confidence, score=100)",
            }
            return {
                "output": answer,
                "user_answer": answer,
                "trace": trace,
                "quality_gate": quality_gate,
                "qa_review": {
                    "verdict": "skipped",
                    "reason": "awaiting user consent for website confirmation",
                },
            }

        answer, trace = self._evaluate_final_answer(user_input=user_input, answer=answer, trace=trace)

        if self._verbose:
            for entry in trace:
                if entry["type"] == "action":
                    print(f"Action: {entry['tool']} | Input: {entry['tool_input']}")
                if entry["type"] == "observation":
                    print(f"Observation ({entry['tool']}): {entry['observation']}")

        quality_gate = _extract_quality_gate(trace)
        qa_review = _extract_qa_review(trace) or {
            "verdict": "skipped",
            "reason": "evaluator output not available",
        }
        user_answer = self._build_user_friendly_answer(
            user_input=user_input,
            technical_answer=answer,
            trace=trace,
            quality_gate=quality_gate,
        )
        user_answer = self._maybe_add_engagement_followup(
            user_input=user_input,
            user_answer=user_answer,
            trace=trace,
        )
        return {
            "output": answer,
            "user_answer": user_answer,
            "trace": trace,
            "quality_gate": quality_gate,
            "qa_review": qa_review,
        }

    def _build_user_friendly_answer(
        self,
        user_input: str,
        technical_answer: str,
        trace: list[dict[str, Any]],
        quality_gate: dict[str, Any],
    ) -> str:
        next_primary = _extract_observation_by_tool(trace, "get_next_launch")
        next_external = _extract_observation_by_tool(trace, "get_next_launch_external")
        latest_primary = _extract_observation_by_tool(trace, "get_latest_launch")
        latest_external = _extract_observation_by_tool(trace, "get_latest_launch_external")
        failed_launch = _extract_observation_by_tool(trace, "get_last_failed_launch")
        launches_in_year = _extract_list_observation_by_tool(trace, "get_launches_in_year")

        requested_year = _extract_year_from_text(user_input)

        if requested_year is not None and launches_in_year is not None:
            launch_count = len(launches_in_year)
            if launch_count == 0:
                return f"I did not find any SpaceX launches in {requested_year}."

            sample_names = [
                str(item.get("name", "a mission"))
                for item in launches_in_year[:3]
                if isinstance(item, dict)
            ]
            examples = ", ".join(sample_names)
            if launch_count == 1:
                return f"I found 1 SpaceX launch in {requested_year}: {examples}."
            return (
                f"I found {launch_count} SpaceX launches in {requested_year}. "
                f"Examples include {examples}."
            )

        if failed_launch and not failed_launch.get("error"):
            name = failed_launch.get("name", "the mission")
            date_utc = _friendly_utc(failed_launch.get("date_utc"))
            details = failed_launch.get("details") or "No additional details available."
            return (
                f"The most recent SpaceX launch failure was {name} on {date_utc}. "
                f"Details: {details}"
            )

        if _is_next_launch_query(user_input):
            # Prefer validated secondary upcoming source when available.
            if next_external and not next_external.get("error"):
                name = next_external.get("name", "the next SpaceX mission")
                date_utc = _friendly_utc(next_external.get("date_utc"))
                status = next_external.get("status", "status not available")
                location = next_external.get("location")
                primary_stale = _has_failed_check(trace, "next_launch_future_guard")
                if location:
                    if primary_stale:
                        return (
                            "The SpaceX API appears to be out of date right now, so I used a backup live source. "
                            f"The next launch I found is {name}, scheduled for {date_utc}, "
                            f"with status {status}, from {location}."
                        )
                    return (
                        f"The next SpaceX launch is {name}, currently scheduled for {date_utc}. "
                        f"Current mission status is {status}, and it is planned from {location}."
                    )
                if primary_stale:
                    return (
                        "The SpaceX API appears to be out of date right now, so I used a backup live source. "
                        f"The next launch I found is {name}, scheduled for {date_utc}, with status {status}."
                    )
                return (
                    f"The next SpaceX launch is {name}, currently scheduled for {date_utc}. "
                    f"Current mission status is {status}."
                )

            # If not validated, still show findings instead of returning nothing useful.
            if next_primary and not next_primary.get("error"):
                primary_date = _parse_iso_utc(next_primary.get("date_utc"))
                if primary_date and primary_date < datetime.now(timezone.utc):
                    return (
                        "Here is what I found: "
                        f"{next_primary.get('name', 'The mission')} is listed for "
                        f"{_friendly_utc(next_primary.get('date_utc'))}. "
                        "I checked additional live sources, but could not confirm a newer launch. "
                        "That date is in the past, so this is likely outdated and probably not the result you wanted."
                    )
                return (
                    "Here is what I found: "
                    f"{next_primary.get('name', 'The mission')} is listed for "
                    f"{_friendly_utc(next_primary.get('date_utc'))}. "
                    "I checked additional live sources, but could not confirm a newer launch. "
                    "Please treat this as unconfirmed until fresh data is available."
                )

            return (
                "I could not find a reliable next-launch date right now. "
                "Please try again in a little while and I will check again."
            )

        if _is_latest_launch_query(user_input):
            if latest_external and not latest_external.get("error"):
                name = latest_external.get("name", "the latest SpaceX mission")
                date_utc = _friendly_utc(latest_external.get("date_utc"))
                status = latest_external.get("status", "status not available")
                primary_stale = _has_failed_check(trace, "latest_launch_freshness")
                if primary_stale:
                    return (
                        "The SpaceX API appears to be out of date right now, so I used a backup live source. "
                        f"The most recent launch I found was {name} on {date_utc}. Mission status: {status}."
                    )
                return f"The most recent SpaceX launch was {name} on {date_utc}. Mission status: {status}."

            if latest_primary and not latest_primary.get("error"):
                return (
                    "Here is what I found: "
                    f"{latest_primary.get('name', 'The mission')} on "
                    f"{_friendly_utc(latest_primary.get('date_utc'))}. "
                    "I checked additional live sources, but could not confirm a newer launch. "
                    "This result is in the past and may be outdated, so it may not be the answer you expected."
                )

            return (
                "I could not find a reliable latest-launch result right now. "
                "Please try again in a little while and I will check again."
            )

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

    def _validate_latest_launch_answer(self, answer: str, trace: list[dict[str, Any]], user_input: str = "") -> tuple[str, list[dict[str, Any]]]:
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

        trace.append(
            {
                "type": "action",
                "tool": "get_latest_launch_external",
                "tool_input": {"strategy": "ll2_then_rocketlaunch_live"},
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

        if external_data and not external_data.get("error"):
            external_date = _parse_iso_utc(external_data.get("date_utc"))
            if external_date and external_date > primary_date:
                _add_determination(
                    trace,
                    check="latest_launch_external_newer",
                    verdict="pass",
                    rationale="Secondary non-website sources returned a newer launch date than primary.",
                    source=external_data.get("source"),
                )
                source_name = _source_label(external_data.get("source"))
                _add_determination(
                    trace,
                    check="website_lookup_consent",
                    verdict="pending",
                    rationale=(
                        "A secondary non-SpaceX source produced a usable answer; waiting for user consent "
                        "to confirm via SpaceX website."
                    ),
                )
                trace.append(
                    {
                        "type": "action",
                        "tool": "website_lookup_consent_prompt",
                        "tool_input": {
                            "query_type": "latest",
                            "prompt": "I found an answer from a non-SpaceX API source. Would you like me to confirm it on the SpaceX website?",
                        },
                    }
                )
                self._pending_website_lookup = {
                    "query_type": "latest",
                    "primary_data": latest_launch,
                    "proposed_data": external_data,
                    "original_query": user_input,
                }
                prompt = (
                    "The SpaceX API appears to be out of date right now. "
                    f"I found a newer latest-launch result from {source_name}: "
                    f"{external_data.get('name')} on {_friendly_utc(external_data.get('date_utc'))} "
                    f"(status: {external_data.get('status')}). "
                    "This is not from the SpaceX API. Would you like me to confirm this by checking the SpaceX website?"
                )
                return prompt, trace

            _add_determination(
                trace,
                check="latest_launch_external_newer",
                verdict="failed",
                rationale="Secondary non-website sources did not provide a newer launch.",
            )

        _add_determination(
            trace,
            check="website_lookup_consent",
            verdict="pending",
            rationale="Primary latest-launch data appears stale; waiting for user consent before searching SpaceX website.",
        )
        trace.append(
            {
                "type": "action",
                "tool": "website_lookup_consent_prompt",
                "tool_input": {
                    "query_type": "latest",
                    "prompt": "I was not able to get the information you requested from the primary source. I can check the SpaceX website. Would you like me to search there?",
                },
            }
        )
        self._pending_website_lookup = {
            "query_type": "latest",
            "primary_data": latest_launch,
            "original_query": user_input,
        }
        consent_prompt = (
            "I was not able to get the information you requested from the primary source. "
            "I could look to see if it is available on the SpaceX website. "
            "Would you like me to search there?"
        )
        return consent_prompt, trace

    def _validate_next_launch_answer(self, answer: str, trace: list[dict[str, Any]], user_input: str = "") -> tuple[str, list[dict[str, Any]]]:
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

        trace.append(
            {
                "type": "action",
                "tool": "get_next_launch_external",
                "tool_input": {"strategy": "ll2_then_rocketlaunch_live"},
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

        if external_data and not external_data.get("error"):
            external_date = _parse_iso_utc(external_data.get("date_utc"))
            if external_date and external_date >= now_utc:
                _add_determination(
                    trace,
                    check="next_launch_external_future_guard",
                    verdict="pass",
                    rationale="Secondary non-website sources returned a future launch date.",
                    source=external_data.get("source"),
                )
                source_name = _source_label(external_data.get("source"))
                _add_determination(
                    trace,
                    check="website_lookup_consent",
                    verdict="pending",
                    rationale=(
                        "A secondary non-SpaceX source produced a usable answer; waiting for user consent "
                        "to confirm via SpaceX website."
                    ),
                )
                trace.append(
                    {
                        "type": "action",
                        "tool": "website_lookup_consent_prompt",
                        "tool_input": {
                            "query_type": "next",
                            "prompt": "I found an answer from a non-SpaceX API source. Would you like me to confirm it on the SpaceX website?",
                        },
                    }
                )
                self._pending_website_lookup = {
                    "query_type": "next",
                    "primary_data": next_launch,
                    "proposed_data": external_data,
                    "original_query": user_input,
                }
                prompt = (
                    "The SpaceX API appears to be out of date right now. "
                    f"I found the next-launch result from {source_name}: "
                    f"{external_data.get('name')} scheduled for {_friendly_utc(external_data.get('date_utc'))} "
                    f"(status: {external_data.get('status')}). "
                    "This is not from the SpaceX API. Would you like me to confirm this by checking the SpaceX website?"
                )
                return prompt, trace

            _add_determination(
                trace,
                check="next_launch_external_future_guard",
                verdict="failed",
                rationale="Secondary non-website sources did not provide a future launch date.",
            )

        _add_determination(
            trace,
            check="website_lookup_consent",
            verdict="pending",
            rationale="Primary next-launch data appears stale; waiting for user consent before searching SpaceX website.",
        )
        trace.append(
            {
                "type": "action",
                "tool": "website_lookup_consent_prompt",
                "tool_input": {
                    "query_type": "next",
                    "prompt": "I was not able to get the information you requested from the primary source. I can check the SpaceX website. Would you like me to search there?",
                },
            }
        )
        self._pending_website_lookup = {
            "query_type": "next",
            "primary_data": next_launch,
            "original_query": user_input,
        }
        consent_prompt = (
            "I was not able to get the information you requested from the primary source. "
            "I could look to see if it is available on the SpaceX website. "
            "Would you like me to search there?"
        )
        return consent_prompt, trace

    def _validate_year_based_answer(self, answer: str, trace: list[dict[str, Any]], user_input: str = "") -> tuple[str, list[dict[str, Any]]]:
        year_based_launches = _extract_list_observation_by_tool(trace, "get_launches_in_year")
        requested_year = _extract_year_from_text(user_input)
        if not requested_year:
            return answer, trace

        if not year_based_launches or len(year_based_launches) == 0:
            _add_determination(
                trace,
                check="year_based_launch_availability",
                verdict="fail",
                rationale=f"Primary source returned no launches for {requested_year}.",
            )
            self._pending_website_lookup = {
                "query_type": "year",
                "primary_data": {},
                "original_query": user_input,
            }
            consent_prompt = (
                f"I was not able to find any SpaceX launches for {requested_year} in the primary source. "
                "I could look to see if it is available on the SpaceX website. "
                "Would you like me to search there?"
            )
            return consent_prompt, trace

        _add_determination(
            trace,
            check="year_based_launch_availability",
            verdict="pass",
            rationale=f"Primary source returned {len(year_based_launches)} launches for {requested_year}.",
        )
        return answer, trace

    def _validate_mission_specific_answer(self, answer: str, trace: list[dict[str, Any]], user_input: str = "") -> tuple[str, list[dict[str, Any]]]:
        # Check if the answer is just asking for confirmation without providing actual data
        answer_lower = answer.lower()
        is_confirmation_prompt = (
            "please reply yes or no" in answer_lower or
            ("would you like me to search" in answer_lower and "website" in answer_lower)
        )

        if is_confirmation_prompt:
            _add_determination(
                trace,
                check="mission_specific_data_availability",
                verdict="fail",
                rationale="Agent could not find mission-specific data in primary source; offering website confirmation.",
            )
            self._pending_website_lookup = {
                "query_type": "mission",
                "primary_data": {},
                "original_query": user_input,
            }
            consent_prompt = (
                "I was not able to find this information in the primary source. "
                "I could look to see if it is available on the SpaceX website. "
                "Would you like me to search there?"
            )
            return consent_prompt, trace

        _add_determination(
            trace,
            check="mission_specific_data_availability",
            verdict="pass",
            rationale="Agent provided mission-specific data from primary source.",
        )
        return answer, trace

    def _evaluate_final_answer(
        self,
        user_input: str,
        answer: str,
        trace: list[dict[str, Any]],
        allow_answer_override: bool = True,
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
            if not allow_answer_override:
                _add_determination(
                    trace,
                    check="final_answer_quality_gate",
                    verdict="fail",
                    rationale=(
                        f"Evaluator requested revision (confidence={confidence}, issues={issues}). "
                        "For confirmation turn safety, keeping original answer text."
                    ),
                    confidence=confidence,
                    fallback_used=False,
                    issues_count=len(issues) if isinstance(issues, list) else 0,
                )
                return answer, trace
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
        if not allow_answer_override:
            _add_determination(
                trace,
                check="final_answer_quality_gate",
                verdict="fail",
                rationale=(
                    f"Evaluator failed answer (confidence={confidence}, issues={issues}, "
                    f"recommended_action={recommended_action}). For confirmation turn safety, keeping original answer text."
                ),
                confidence=confidence,
                fallback_used=False,
                issues_count=len(issues) if isinstance(issues, list) else 0,
            )
            return answer, trace
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
