from __future__ import annotations

from src.agent import build_agent_session


INTERVIEW_SAMPLE_QUESTIONS = [
    "When was the last SpaceX launch?",
    "What's the next SpaceX launch and where is it happening?",
    "How many launches did SpaceX complete in 2024?",
    "Which rocket was used for the Starlink 9-1 mission?",
    "Show me all successful Falcon 9 launches.",
    "What was the outcome of the first Falcon Heavy launch?",
    "Tell me about the most recent launch from Vandenberg.",
]


def _print_trace(trace: list[dict]) -> None:
    if not trace:
        print("  (No tool trace captured)")
        return

    current_step = 0
    for item in trace:
        item_type = item.get("type")
        if item_type == "action":
            current_step += 1
            print(f"  Step {current_step} Action: {item.get('tool')}")
            print(f"  Step {current_step} Action Input: {item.get('tool_input')}")
        elif item_type == "observation":
            print(f"  Step {current_step} Observation: {item.get('observation')}")


def run_demo() -> None:
    print("=== SpaceX Agentic Demo Runner ===")
    print("Running interview sample prompts with full trace output...\n")

    session = build_agent_session(verbose=False)

    for index, question in enumerate(INTERVIEW_SAMPLE_QUESTIONS, start=1):
        print(f"Question {index}: {question}")
        result = session.ask(question)
        print(f"Answer {index}: {result.get('output', '')}")
        print("Trace:")
        _print_trace(result.get("trace", []))
        print("-" * 80)


if __name__ == "__main__":
    run_demo()
