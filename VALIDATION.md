# Validation Overview

This document summarizes how the SpaceX agent's accuracy and reliability are validated.

## What Is Validated

1. Conversational context across multiple turns
2. Latest-launch freshness checks (stale data detection)
3. Next-launch future-date checks (reject past "next" dates)
4. Ambiguous input clarification behavior
5. Final answer quality gate metadata (status, confidence, score)

## How Validation Works

- The app includes `src/validation_runner.py` for repeatable validation checks.
- Each check runs a realistic prompt and verifies expected trace behavior and outputs.
- Validation emphasizes grounded responses and non-hallucination safeguards.

## Run Validation

```bash
python -m src.validation_runner
```

## Reliability Safeguards Implemented

1. Dual-source verification for stale latest/next launch scenarios
2. Deterministic guards for stale/future-date consistency
3. Clarification guard for ambiguous launch questions
4. Final-answer evaluator gate before user-facing response
5. Visible Think-Act-Observe trace with explicit determinations

## Notes

- Scores and confidence values are transparent in outputs.
- If evidence is insufficient, the agent returns a safe fallback instead of guessing.
