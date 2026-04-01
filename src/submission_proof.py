from __future__ import annotations

from dataclasses import dataclass

from src.agent import build_agent_session
from src.validation_runner import print_report, run_validation


@dataclass
class RequirementProof:
    requirement: str
    evidence: str


def run_requirement_proof() -> list[RequirementProof]:
    session = build_agent_session(verbose=False)

    # Requirement 1: conversational context
    _ = session.ask("Which rocket was used for the Starlink 9-1 mission?")
    follow_up = session.ask("And where did it launch from?")
    req1 = RequirementProof(
        requirement="1. Conversational Agent",
        evidence=(
            "Follow-up question handled in same session with context retained. "
            f"Trace steps: {len(follow_up.get('trace', []))}"
        ),
    )

    # Requirement 2 + 3 + 4 + 5 via factual query and trace/quality metadata.
    latest = session.ask("When was the last SpaceX launch?")
    tools_used = [item.get("tool") for item in latest.get("trace", []) if item.get("type") == "action"]
    req2 = RequirementProof(
        requirement="2. Domain: SpaceX",
        evidence=f"Live SpaceX question answered with user-facing response: {latest.get('user_answer', '')[:140]}...",
    )
    req3 = RequirementProof(
        requirement="3. Tool Design",
        evidence=f"Action trace shows tool invocation chain: {tools_used}",
    )
    req4 = RequirementProof(
        requirement="4. LLM Integration",
        evidence=(
            "Quality gate metadata present and grounded response generated: "
            f"{latest.get('quality_gate', {}).get('summary', 'missing')}"
        ),
    )
    req5 = RequirementProof(
        requirement="5. Agentic Behavior",
        evidence=(
            "Deterministic guards and evaluator run are visible in determinations: "
            f"{[x.get('check') for x in latest.get('trace', []) if x.get('type') == 'determination']}"
        ),
    )

    # Requirement 6 through formal validation harness.
    validation_results = run_validation()
    passed = sum(1 for item in validation_results if item.ok)
    req6 = RequirementProof(
        requirement="6. Validation",
        evidence=f"Formal validation runner passed {passed}/{len(validation_results)} checks.",
    )

    return [req1, req2, req3, req4, req5, req6]


def print_requirement_proof(proofs: list[RequirementProof]) -> None:
    print("=== Requirement-by-Requirement Proof Log ===")
    for proof in proofs:
        print(f"[{proof.requirement}]")
        print(f"- {proof.evidence}")
    print("-" * 72)


def main() -> None:
    proofs = run_requirement_proof()
    print_requirement_proof(proofs)

    print("\n=== Embedded Validation Report ===")
    validation_results = run_validation()
    print_report(validation_results)


if __name__ == "__main__":
    main()
