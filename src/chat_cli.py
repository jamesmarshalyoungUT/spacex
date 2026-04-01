from __future__ import annotations

from src.agent import build_agent_session


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
        print("\nAgent:", result.get("output", ""))

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
                    print(f"Step {step_idx} Observation: {item.get('observation')}\n")
            print("--- End Trace ---\n")


if __name__ == "__main__":
    run_chat()
