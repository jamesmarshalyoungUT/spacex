# SpaceX Agentic AI Platform

A production-style conversational AI system that answers SpaceX questions using live data, tool-grounded reasoning, and transparent decision traces.

This project combines an orchestrated multi-agent workflow with deterministic guardrails to deliver reliable, explainable answers across CLI, API, and Streamlit interfaces.

## What This Solution Delivers

- Agentic conversation with memory across turns
- Multi-step tool orchestration with visible Think-Act-Observe traces
- SpaceX domain grounding via SpaceX API v5
- Fallback verification pipeline for stale or missing launch data
- Consent-first website confirmation flow before SpaceX website lookup
- Timeout-safe and exception-safe external lookup handling
- Query-type-aware validation behavior (latest/next, year-based, mission-specific)
- Fuzzy typo-tolerant launch intent matching (for example, "lauch" -> "launch")
- Final-answer quality gate with confidence scoring
- Engagement follow-up agent that can execute the next suggested action on user approval

## Stack

- Python
- LangChain ReAct agent
- SpaceX API v5 (`https://api.spacexdata.com/v5`)
- FastAPI (optional API surface)
- Streamlit (interactive app)
- CLI chat interface

## Architecture

```mermaid
flowchart TD
  A[User: CLI / Streamlit / API] --> B[Session Orchestrator]
  B --> C[Reasoning Agent]
  C --> D[SpaceX Tooling Layer]
  D --> E[SpaceX API v5]

  B --> F[Deterministic Guardrails]
  F --> G[Freshness + Future Validation]
  G --> H[External Cross-Checks]
  H --> I[Launch Library 2]
  H --> J[RocketLaunch.Live]

  G --> K[Consent Gate]
  K --> L{User approved website lookup?}
  L -->|Yes| M[SpaceX Website Lookup\n(timeout + error safe)]
  L -->|No| N[Return verified non-website answer]

  B --> O[Final Answer Evaluator]
  O --> P[Quality Gate + Confidence]
  P --> Q[User-Facing Answer]

  Q --> R[Engagement Follow-Up Agent]
  R --> S[Offer next action\n"Yes" executes suggested query]
```

## Guardrail and Consent Behavior

The platform does not silently scrape the SpaceX website. It explains source limitations first, then asks for explicit user approval where website confirmation is supported.

### Latest/Next launch questions

- Uses SpaceX API as primary source.
- If data appears stale (older than 180 days), cross-checks Launch Library 2 and RocketLaunch.Live.
- If needed, requests user consent before website confirmation lookup.

### Year-based launch questions

- Queries the primary source for launches in the requested year.
- If unavailable, explains source limitation.
- Suggests alternative query paths because website confirmation is not enabled for this query class.

### Mission-specific questions

- Attempts mission lookup from primary source and internal tools.
- If unresolved, explains source limitation and offers alternatives.
- Website confirmation is not enabled for this query class.

## QA Evaluator and Follow-Up Agent

Two post-answer agents are intentionally separated:

1. QA Evaluator Agent
- Checks intent match, grounding, and answer quality.
- Produces structured review metadata and confidence.
- Acts as a trust and reliability layer.

2. Engagement Follow-Up Agent
- Offers one relevant next action after each completed response.
- If user replies yes, automatically executes the suggested next query.
- Handles user experience continuity without mixing with safety decisions.

## Project Structure

- `src/spacex_client.py`: SpaceX API wrapper and HTTP handling
- `src/tools.py`: domain tools for launches, rockets, launchpads, and filtering
- `src/agent.py`: ReAct agent setup, orchestration logic, and memory integration
- `src/chat_cli.py`: terminal chat experience with trace output
- `src/api_server.py`: FastAPI app with session-aware chat endpoint
- `src/streamlit_app.py`: Streamlit chat UI with per-turn trace rendering
- `src/demo_script.py`: scripted end-to-end demo conversations
- `src/submission_proof.py`: requirement and validation proof output utility
- `src/validation_runner.py`: formal validation runner
- `src/static/index.html`: lightweight trace timeline viewer

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Configure environment

```bash
copy .env.example .env
```

3. Set your key in `.env`

```env
GEMINI_API_KEY=your_key_here
```

## Run Interfaces

### CLI

```bash
python -m src.chat_cli
```

Example prompts:

- `When was the last SpaceX launch?`
- `What's the next SpaceX launch and where is it happening?`
- `How many launches did SpaceX complete in 2024?`
- `Which rocket was used for the Starlink 9-1 mission?`

### API (FastAPI)

```bash
uvicorn src.api_server:app --reload
```

Open:

- `http://127.0.0.1:8000/`

Endpoints:

- `GET /health`
- `POST /chat`

Example request:

```json
{
  "session_id": "demo-session-1",
  "message": "What was the outcome of the first Falcon Heavy launch?"
}
```

### Streamlit

```bash
streamlit run src/streamlit_app.py
```

For local secrets, create `.streamlit/secrets.toml` and set:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.0-flash"
SPACEX_API_BASE_URL = "https://api.spacexdata.com/v5"
```

## Validation and Operational Confidence

Run validation checks:

```bash
python -m src.validation_runner
```

Reference validation notes in `VALIDATION.md`.

## Observability and Reliability

- Answers are grounded in tool outputs, not free-form generation alone.
- Intermediate reasoning actions are traceable per turn.
- Tooling is intentionally granular so the agent can compose robust multi-step plans.
- Fallback behavior is consent-gated, timeout-safe, and failure-tolerant.

## Demo Utilities

Run scripted conversation demo:

```bash
python -m src.demo_script
```

Run proof/report utility:

```bash
python -m src.submission_proof
```
