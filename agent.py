"""Agent loop with two interchangeable model backends (Gemini and Groq) and automatic fallback.

The model may call tools (RAG search, Wikipedia, pin_song) several times before answering. If the
first backend fails (rate limit, quota, outage, bad model name), the whole turn is retried on the next
configured backend, so a free-tier limit on one provider doesn't take the app down.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from google.genai import types

import observability as obs
import security
from tools import TOOL_SCHEMAS, ToolRunner

MAX_TOOL_STEPS = 4
MAX_OUTPUT_TOKENS = 2048  # Gemini 3.x thinking tokens count toward this limit
GROQ_MAX_TOKENS = 1500
RETRY_CODES = (429, 500, 503)

# Gemini function declarations built from the shared tool schemas.
_GEMINI_TOOL = types.Tool(function_declarations=[
    {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in TOOL_SCHEMAS
])
# The same tools in OpenAI/Groq format.
_GROQ_TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOL_SCHEMAS
]


@dataclass
class AgentResult:
    text: str
    trace: list[dict] = field(default_factory=list)
    provider: str = ""


class AllBackendsFailed(Exception):
    """Every configured backend failed; the message lists each provider's error."""


# ------------------------------------------------------------------ Gemini
def to_contents(messages: list[dict]) -> list[types.Content]:
    """Convert [{"role": "user"|"assistant", "content": str}] to Gemini contents."""
    return [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]


@obs.observe(as_type="generation", name="gemini-call")
def call_model(client, model: str, system: str, contents: list, force_no_tools: bool = False, retries: int = 2):
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[_GEMINI_TOOL],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE" if force_no_tools else "AUTO")
        ),
    )
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) in RETRY_CODES and attempt < retries:
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


class GeminiBackend:
    name = "gemini"

    def __init__(self, client, model: str, retries: int = 2):
        self.client, self.model, self.retries = client, model, retries

    def run(self, system, messages, runner, max_steps=MAX_TOOL_STEPS, on_tool=None) -> AgentResult:
        contents = to_contents(messages)
        trace: list[dict] = []
        for step in range(max_steps + 1):
            last = step == max_steps
            resp = call_model(self.client, self.model, system, contents, force_no_tools=last, retries=self.retries)
            calls = [] if last else (resp.function_calls or [])
            if not calls:
                text = (resp.text or "").strip()
                return AgentResult(text or "Sorry, I couldn't put an answer together. Could you rephrase?", trace, self.name)
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
        return AgentResult("Sorry, I couldn't finish that lookup. Could you try again?", trace, self.name)


# ------------------------------------------------------------------ Groq (OpenAI-compatible API)
@obs.observe(as_type="generation", name="groq-call")
def call_groq(client, model: str, messages: list, force_no_tools: bool = False):
    kwargs = dict(model=model, messages=messages, tools=_GROQ_TOOLS, max_tokens=GROQ_MAX_TOKENS,
                  tool_choice="none" if force_no_tools else "auto")
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        if "tool_use_failed" in str(e) and not force_no_tools:
            # Llama occasionally emits a malformed tool call; retry once without tools.
            kwargs["tool_choice"] = "none"
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    usage = getattr(resp, "usage", None)
    obs.record_generation(
        model=model,
        usage_details={"input": getattr(usage, "prompt_tokens", 0) or 0,
                       "output": getattr(usage, "completion_tokens", 0) or 0},
    )
    return resp


class GroqBackend:
    name = "groq"

    def __init__(self, client, model: str):
        self.client, self.model = client, model

    def run(self, system, messages, runner, max_steps=MAX_TOOL_STEPS, on_tool=None) -> AgentResult:
        convo = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
        trace: list[dict] = []
        for step in range(max_steps + 1):
            last = step == max_steps
            msg = call_groq(self.client, self.model, convo, force_no_tools=last).choices[0].message
            calls = [] if last else (getattr(msg, "tool_calls", None) or [])
            if not calls:
                text = (msg.content or "").strip()
                return AgentResult(text or "Sorry, I couldn't put an answer together. Could you rephrase?", trace, self.name)
            convo.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                               for tc in calls],
            })
            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_tool:
                    on_tool(tc.function.name, args)
                output = runner.run(tc.function.name, args if isinstance(args, dict) else {})
                trace.append({"tool": tc.function.name, "input": args, "output_preview": output[:300]})
                convo.append({"role": "tool", "tool_call_id": tc.id, "content": output})
        return AgentResult("Sorry, I couldn't finish that lookup. Could you try again?", trace, self.name)


# ------------------------------------------------------------------ orchestration
@obs.observe(name="songexplainer-agent")
def run_agent(backends: list, system: str, messages: list[dict], runner: ToolRunner,
              max_steps: int = MAX_TOOL_STEPS, on_tool=None) -> AgentResult:
    """Try each backend in order; return the first successful result."""
    errors: list[str] = []
    for backend in backends:
        try:
            return backend.run(system, messages, runner, max_steps=max_steps, on_tool=on_tool)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{backend.name}: {type(e).__name__}: {str(e)[:200]}")
    raise AllBackendsFailed(" | ".join(errors))


@obs.observe(name="songexplainer-turn")
def answer_turn(backends: list, system: str, messages: list[dict], runner: ToolRunner, canary: str,
                meta: dict | None = None, session_id: str | None = None,
                on_tool=None) -> tuple[str, list[str], list[dict], str]:
    """One full chat turn: agent loop -> output guard -> trace metadata.
    Returns (text, guard_flags, tool_trace, provider_that_answered)."""
    result = run_agent(backends, system, messages, runner, on_tool=on_tool)
    text, flags = security.guard_output(result.text, canary, system)
    obs.update_trace(session_id=session_id, metadata={**(meta or {}), "guard_flags": flags,
                                                      "provider": result.provider,
                                                      "tools_used": [t["tool"] for t in result.trace]})
    return text, flags, result.trace, result.provider
