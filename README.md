# SpaceX Agentic Hiring Test Solution

This project implements a conversational AI agent for SpaceX questions using live SpaceX API data.

It is built to demonstrate **agentic behavior** required by the challenge:
- Conversational memory across turns
- Tool-based factual grounding
- Multi-step reasoning with multiple tool calls
- Clarifying questions for ambiguous input
- Error handling and graceful fallback
- Visible Think-Act-Observe trace (not a single input-output line)
- Deterministic freshness/future validation guards for stale launch data
- Fuzzy typo-tolerant intent matching for key launch keywords (e.g., "lauch" -> "launch")
- Explicit user-consent gate before SpaceX website lookup
- Timeout-safe and exception-safe website fallback handling
- Engagement follow-up agent that offers one actionable next step; saying yes executes it

## Stack

- Python
- LangChain ReAct agent
- SpaceX v5 API (`https://api.spacexdata.com/v5`)
- CLI chat interface
- Optional FastAPI web endpoint
- Streamlit demo app

## Architecture Diagram

```mermaid
flowchart TD
  A[User CLI / Streamlit / API] --> B[SpaceX Session Orchestrator Agent]
  B --> C[Primary Reasoning Agent]
  C --> D[SpaceX Tools]
  D --> E[SpaceX API v5]
  B --> F[Deterministic Guards]
  F --> G[Freshness/Future Checks]
  G --> L[External Cross-Check Tools (non-agent)]
  L --> M[Launch Library 2]
  L --> O[RocketLaunch.Live]
  L --> H[Consent Prompt for SpaceX Website Confirmation]
  H --> I{User says yes?}
  I -->|Yes| J[SpaceX Website Lookup\n(timeout + error handling)]
  I -->|No| K[Keep non-SpaceX verified answer]
  B --> P[Final Answer Evaluator Agent]
  P --> Q[Quality Gate + Confidence Score]
  P --> T[QA Evaluator Agent Review JSON]
  Q --> R[Friendly User Answer]
  R --> U[Engagement Follow-Up Agent\n(offer + yes executes action)]
  Q --> S[Think-Act-Observe Trace + Determinations]
```

## Guard Behavior (Consent Before Website Search)

When stale primary launch data is detected, the agent does not automatically scrub the SpaceX website.

It asks the user first with this wording:

"I was not able to get the information you requested from the primary source. I could look to see if it is available on the SpaceX website. Would you like me to search there?"

Behavior:
- If user says yes: agent performs website lookup.
- If user says no: agent skips website lookup and keeps the already verified non-SpaceX source answer.
- If response is unclear: agent asks for a clear yes/no.
- If website lookup times out or errors: agent returns a safe fallback message and does not crash.

Current lookup order for stale latest/next launch cases:
1. SpaceX API v5 (primary)
2. Launch Library 2 and RocketLaunch.Live (secondary cross-check)
3. SpaceX website confirmation only after explicit user consent

## QA Evaluator and Engagement Follow-Up Agent

This system intentionally uses two different post-answer agents because they serve different goals.

### 1) QA Evaluator Agent (Accuracy + Trust)

Purpose:
- Validate that the final answer matches user intent.
- Validate that the answer is grounded in observed tool outputs.
- Return a structured verdict with confidence and issues.

What it produces:
- `qa_review` JSON (intent match, grounded-in-trace check, pass/fail verdict, confidence, issues, recommended action)
- `quality_gate` metadata (status + confidence score) shown in CLI/Streamlit/API

Why it exists:
- Keeps answers reliable and auditable.
- Makes evaluation explicit for interview reviewers.
- Prevents silent hallucinations by forcing trace-grounded checks.

### 2) Engagement Follow-Up Agent (Conversation Continuation)

Purpose:
- Offer one optional, actionable next step after every completed answer.
- Encourage continued interaction; if the user says yes, the agent actually executes the suggested action.
- Runs on all completed turns, including decline ("no") responses from the website consent gate.

What it produces:
- `offer_text`: short description of what the agent will look up (e.g., "I can find details about the rocket used in this mission")
- `suggested_query`: the natural-language question executed when user says yes
- A prompt appended to user-facing output: `"If you're interested, I can <offer_text>. Would you like me to do that?"`
- Session state `_pending_engagement_action` stores the suggested query; a yes/no response on the next turn either executes it or dismisses it

Why it exists:
- Improves user retention and session depth.
- Keeps prompts domain-relevant (SpaceX context), not generic filler.
- Separates engagement behavior from factual validation logic.

### Why these are separate agents

- QA Evaluator is a safety/quality function.
- Engagement Follow-Up is a UX/conversation function.
- Keeping them separate avoids mixing trust decisions with engagement goals.
- This separation also makes traces easier to audit during interviews.

### Example Trace Snippet

```text
Step 3 (P-Q) Action: final_answer_evaluator_agent
Step 3 (P-Q) Observation: {"verdict":"pass","confidence":"high",...}
Step 3 (P-Q) Determination: final_answer_quality_gate => pass

Step 4 (A-B) Action: engagement_followup_agent
Step 4 (A-B) Observation: {"include_followup":true,"offer_text":"I can find details about the Falcon 9 rocket","suggested_query":"What rocket type is used for the next SpaceX launch?"}
Step 4 (A-B) Determination: engagement_followup => pass
```

## Project Files

- `src/spacex_client.py`: robust SpaceX API wrapper
- `src/tools.py`: agent tools (launch/rocket/launchpad search, year queries, etc.)
- `src/agent.py`: ReAct prompt + executor + memory
- `src/chat_cli.py`: terminal chatbot with reasoning trace output
- `src/api_server.py`: optional HTTP interface with per-session memory
- `src/demo_script.py`: runs exact interview sample questions automatically with traces
- `src/static/index.html`: lightweight web page to visualize Action/Observation timeline
- `src/streamlit_app.py`: Streamlit chat app with per-turn trace visualization

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

Then edit `.env` and set `GEMINI_API_KEY`.

## Run CLI (recommended for interview demo)

```bash
python -m src.chat_cli
```

Sample prompts:
- `When was the last SpaceX launch?`
- `What's the next SpaceX launch and where is it happening?`
- `How many launches did SpaceX complete in 2024?`
- `Which rocket was used for the Starlink 9-1 mission?`
- `Show me all successful Falcon 9 launches.`
- `Tell me about the most recent launch from Vandenberg.`

The CLI prints:
- Final answer
- Detailed Think-Act-Observe tool trace per turn

## Run Auto Demo Script

```bash
python -m src.demo_script
```

This runs the exact sample prompts from the hiring prompt and prints:
- Question
- Final answer
- Step-by-step Action and Observation trace

## Run Submission Proof Log (One Command)

```bash
python -m src.submission_proof
```

This prints a requirement-by-requirement proof log and embeds the formal validation report.

## Run API Server (optional)

```bash
uvicorn src.api_server:app --reload
```

Then open:
- `http://127.0.0.1:8000/`

Endpoints:
- `GET /health`
- `POST /chat`

Example request body:

```json
{
  "session_id": "candidate-demo-1",
  "message": "What was the outcome of the first Falcon Heavy launch?"
}
```

Response includes `answer` and a `trace` array containing tool calls and observations.

## Run Streamlit App

```bash
streamlit run src/streamlit_app.py
```

### Streamlit Secrets (Recommended)

For local Streamlit development:

1. Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.
2. Add your real keys in `.streamlit/secrets.toml`.
3. Keep `.streamlit/secrets.toml` out of git (already ignored in `.gitignore`).

Example:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.0-flash"
SPACEX_API_BASE_URL = "https://api.spacexdata.com/v5"
```

For Streamlit Community Cloud:

1. Open your app settings.
2. Go to `Secrets`.
3. Paste the same TOML content and save.

The app reads secrets with `st.secrets` and uses them at runtime.

The Streamlit app provides:
- conversational chat
- session reset button
- interview sample question picker
- Action/Observation trace visualization per turn

## Validation

Run formal validation checks:

```bash
python -m src.validation_runner
```

See full validation notes in `VALIDATION.md`.

## Requirement Coverage

1. Conversational Agent:
- CLI, Streamlit, and API chat interfaces
- session memory maintained across turns

2. Domain: SpaceX:
- live data answers across launch, rocket, launchpad, and location queries

3. Tool Design:
- dedicated tools for API calls, parsing, filtering, and fallback handling

4. LLM Integration:
- intent interpretation + tool orchestration + grounded answer synthesis

5. Agentic Behavior:
- multi-tool reasoning loops
- deterministic freshness/future guards
- clarifying questions for ambiguous launch input
- explicit consent prompt before website lookup when primary data is stale
- timeout-safe and exception-safe website lookup execution
- final-answer evaluator agent with quality gate

6. Validation:
- repeatable validation runner and report (`src/validation_runner.py`, `VALIDATION.md`)

## Beyond The Ask

- Quality Gate metadata with confidence score (0-100)
- Friendly customer-facing answers separated from technical trace
- Friendly date formatting in user answers and trace hints
- External cross-source verification to mitigate stale primary API data
- SpaceX website direct fallback source integrated in addition to Launch Library 2 and RocketLaunch.Live
- Consent-first fallback policy to avoid unexpected web scraping behavior

## Notes for Interviewers

- Answers are grounded in real tool outputs from SpaceX API.
- Agent behavior is observable via verbose traces and returned intermediate steps.
- The design intentionally keeps tools granular so the agent can chain them autonomously.
