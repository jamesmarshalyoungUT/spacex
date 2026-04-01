from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import SPACEX_TOOLS


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


class SpaceXAgentSession:
    def __init__(self, model_name: str, temperature: float = 0, verbose: bool = True) -> None:
        self._verbose = verbose
        self._messages: list[Any] = []
        self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self._graph = create_react_agent(model=self._llm, tools=SPACEX_TOOLS, prompt=SYSTEM_PROMPT)

    def ask(self, user_input: str) -> dict[str, Any]:
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

        if self._verbose:
            for entry in trace:
                if entry["type"] == "action":
                    print(f"Action: {entry['tool']} | Input: {entry['tool_input']}")
                if entry["type"] == "observation":
                    print(f"Observation ({entry['tool']}): {entry['observation']}")

        return {"output": answer, "trace": trace}


def build_agent_session(verbose: bool = True) -> SpaceXAgentSession:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in .env before running the agent.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return SpaceXAgentSession(model_name=model_name, temperature=0, verbose=verbose)
