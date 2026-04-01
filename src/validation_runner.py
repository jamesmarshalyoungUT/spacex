from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agent import build_agent_session


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _has_trace_type(trace: list[dict], trace_type: str) -> bool:
    return any(item.get("type") == trace_type for item in trace)


def _has_tool(trace: list[dict], tool_name: str) -> bool:
    return any(item.get("tool") == tool_name for item in trace)


def _run_check(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, ok=False, details=f"Exception: {exc}")


def run_validation() -> list[CheckResult]:
    session = build_agent_session(verbose=False)
    results: list[CheckResult] = []

    def check_conversational_context() -> CheckResult:
        session.ask("Which rocket was used for the Starlink 9-1 mission?")
        second = session.ask("And where did it launch from?")
        ok = bool(second.get("user_answer")) and _has_trace_type(second.get("trace", []), "action")
        return CheckResult(
            name="Conversational context maintained",
            ok=ok,
            details="Follow-up question processed with trace activity.",
        )

    def check_latest_freshness_guard() -> CheckResult:
        res = session.ask("When was the last SpaceX launch?")
        trace = res.get("trace", [])
        ok = _has_tool(trace, "freshness_guard") and _has_tool(trace, "get_latest_launch_external")
        return CheckResult(
            name="Latest launch freshness validation",
            ok=ok,
            details="Expected freshness guard and external cross-check in trace.",
        )

    def check_next_future_guard() -> CheckResult:
        res = session.ask("When is the next launch?")
        trace = res.get("trace", [])
        ok = _has_tool(trace, "future_guard") and _has_tool(trace, "get_next_launch_external")
        return CheckResult(
            name="Next launch future-date validation",
            ok=ok,
            details="Expected future guard and external cross-check in trace.",
        )

    def check_ambiguity_clarification() -> CheckResult:
        res = session.ask("Tell me about launch")
        trace = res.get("trace", [])
        ok = _has_tool(trace, "clarification_guard") and "latest launch" in res.get("user_answer", "").lower()
        return CheckResult(
            name="Ambiguous input clarification",
            ok=ok,
            details="Expected clarification guard and clarifying question output.",
        )

    def check_quality_gate_output() -> CheckResult:
        res = session.ask("How many launches did SpaceX complete in 2024?")
        qg = res.get("quality_gate", {})
        ok = all(k in qg for k in ["status", "confidence", "confidence_score", "summary"])
        return CheckResult(
            name="Quality gate metadata present",
            ok=ok,
            details=f"quality_gate={qg}",
        )

    checks = [
        ("Conversational context maintained", check_conversational_context),
        ("Latest launch freshness validation", check_latest_freshness_guard),
        ("Next launch future-date validation", check_next_future_guard),
        ("Ambiguous input clarification", check_ambiguity_clarification),
        ("Quality gate metadata present", check_quality_gate_output),
    ]

    for name, fn in checks:
        results.append(_run_check(name, fn))

    return results


def print_report(results: list[CheckResult]) -> None:
    print("=== Validation Report ===")
    passed = 0
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        if item.ok:
            passed += 1
        print(f"[{status}] {item.name}")
        print(f"  {item.details}")

    print("-" * 72)
    print(f"Summary: {passed}/{len(results)} checks passed")


if __name__ == "__main__":
    report = run_validation()
    print_report(report)
