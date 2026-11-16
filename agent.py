"""Agent loop (Gemini): the model may call tools (RAG search, Wikipedia, pin_song) several times
before answering. Uses the google-genai SDK with manual function calling."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from google.genai import types

import observability as obs
import security
from tools import TOOL_SCHEMAS, ToolRunner

MAX_TOOL_STEPS = 4
MAX_OUTPUT_TOKENS = 2048  # Gemini 3.x thinking tokens count toward this limit
RETRY_CODES = (429, 500, 503)

# Gemini function declarations built from the shared tool schemas.
_TOOL = types.Tool(function_declarations=[
    {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in TOOL_SCHEMAS
])


@dataclass
class AgentResult:
    text: str
    trace: list[dict] = field(default_factory=list)


def to_contents(messages: list[dict]) -> list[types.Content]:
    """Convert [{"role": "user"|"assistant", "content": str}] to Gemini contents."""
    return [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]


@obs.observe(as_type="generation", name="gemini-call")
def call_model(client, model: str, system: str, contents: list, force_no_tools: bool = False):
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[_TOOL],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE" if force_no_tools else "AUTO")
        ),
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) in RETRY_CODES and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    usage = getattr(resp, "usage_metadata", None)
    obs.record_generation(
        model=model,
        usage_details={"input": getattr(usage, "prompt_token_count", 0) or 0,
                       "output": getattr(usage, "candidates_token_count", 0) or 0},
    )
    return resp


@obs.observe(name="songexplainer-agent")
def run_agent(client, model: str, system: str, messages: list[dict], runner: ToolRunner,
              max_steps: int = MAX_TOOL_STEPS, on_tool=None) -> AgentResult:
    contents = to_contents(messages)
    trace: list[dict] = []
    for step in range(max_steps + 1):
        last = step == max_steps
        resp = call_model(client, model, system, contents, force_no_tools=last)
        calls = [] if last else (resp.function_calls or [])
        if not calls:
            text = (resp.text or "").strip()
            return AgentResult(text or "Sorry, I couldn't put an answer together. Could you rephrase?", trace)
        # Keep the model's turn exactly as returned (preserves Gemini 3 thought signatures).
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            if on_tool:
                on_tool(fc.name, args)
            output = runner.run(fc.name, args)
            trace.append({"tool": fc.name, "input": args, "output_preview": output[:300]})
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": output}))
        contents.append(types.Content(role="user", parts=parts))
    return AgentResult("Sorry, I couldn't finish that lookup. Could you try again?", trace)


@obs.observe(name="songexplainer-turn")
def answer_turn(client, model: str, system: str, messages: list[dict], runner: ToolRunner, canary: str,
                meta: dict | None = None, session_id: str | None = None, on_tool=None) -> tuple[str, list[str], list[dict]]:
    """One full chat turn: agent loop -> output guard -> trace metadata. Returns (text, flags, tool_trace)."""
    result = run_agent(client, model, system, messages, runner, on_tool=on_tool)
    text, flags = security.guard_output(result.text, canary, system)
    obs.update_trace(session_id=session_id, metadata={**(meta or {}), "guard_flags": flags,
                                                      "tools_used": [t["tool"] for t in result.trace]})
    return text, flags, result.trace
