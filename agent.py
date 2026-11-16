"""Agent loop: the model may call tools (RAG search, Wikipedia) several times before answering."""
from __future__ import annotations

from dataclasses import dataclass, field

import observability as obs
import security
from tools import TOOL_SCHEMAS, ToolRunner

MAX_TOOL_STEPS = 4
MAX_TOKENS = 900


@dataclass
class AgentResult:
    text: str
    trace: list[dict] = field(default_factory=list)


@obs.observe(as_type="generation", name="claude-call")
def call_model(client, model: str, system: str, messages: list, force_no_tools: bool = False):
    kwargs = {"tool_choice": {"type": "none"}} if force_no_tools else {}
    resp = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system, messages=messages, tools=TOOL_SCHEMAS, **kwargs
    )
    usage = getattr(resp, "usage", None)
    obs.record_generation(
        model=model,
        usage_details={"input": getattr(usage, "input_tokens", 0), "output": getattr(usage, "output_tokens", 0)},
    )
    return resp


@obs.observe(name="songexplainer-agent")
def run_agent(client, model: str, system: str, messages: list[dict], runner: ToolRunner,
              max_steps: int = MAX_TOOL_STEPS, on_tool=None) -> AgentResult:
    convo = list(messages)
    trace: list[dict] = []
    for step in range(max_steps + 1):
        last = step == max_steps
        resp = call_model(client, model, system, convo, force_no_tools=last)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            return AgentResult(text or "Sorry, I couldn't put an answer together. Could you rephrase?", trace)
        convo.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            if on_tool:
                on_tool(block.name, block.input)
            output = runner.run(block.name, block.input)
            trace.append({"tool": block.name, "input": block.input, "output_preview": output[:300]})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        convo.append({"role": "user", "content": results})
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
