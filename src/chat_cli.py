from __future__ import annotations

import json
from datetime import datetime, timezone

from src.agent import build_agent_session


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


def run_chat() -> None:
    print("SpaceX Agentic Chat")
    print("Type 'exit' to quit.\n")

    session = build_agent_session(verbose=False)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = session.ask(user_input)
        user_answer = result.get("user_answer", result.get("output", ""))
        quality_gate = result.get("quality_gate", {})
        status = str(quality_gate.get("status", "unknown")).upper()
        confidence = str(quality_gate.get("confidence", "low")).lower()
        score = quality_gate.get("confidence_score", 50)
        fallback_used = bool(quality_gate.get("fallback_used", False))

        print("\nAI:", user_answer)
        print("\n" + "=" * 72)
        print("Behind The Scenes")
        print("=" * 72)

        if fallback_used:
            print(f"Quality Gate: {status} ({confidence} confidence, score={score}, fallback used)")
        else:
            print(f"Quality Gate: {status} ({confidence} confidence, score={score})")

        steps = result.get("trace", [])
        if steps:
            print("\n--- Think-Act-Observe Trace ---")
            step_idx = 0
            for item in steps:
                if item.get("type") == "action":
                    step_idx += 1
                    print(f"Step {step_idx} Action: {item.get('tool')}")
                    print(f"Step {step_idx} Action Input: {item.get('tool_input')}")
                elif item.get("type") == "observation":
                    observation_text = str(item.get("observation"))
                    print(f"Step {step_idx} Observation: {observation_text}")
                    hints = _friendly_date_hints(observation_text)
                    if hints:
                        print(f"Step {step_idx} Friendly Dates:")
                        for path, raw_value, friendly in hints:
                            print(f"  - {path}: {raw_value} -> {friendly}")
                    print("")
                elif item.get("type") == "determination":
                    print(
                        f"Step {step_idx} Determination: "
                        f"{item.get('check')} => {item.get('verdict')}"
                    )
                    print(f"Step {step_idx} Rationale: {item.get('rationale')}\n")
            print("--- End Trace ---\n")


if __name__ == "__main__":
    run_chat()
