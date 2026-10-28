#!/usr/bin/env python3
"""The same pipeline rebuilt on the OpenAI Agents SDK, running the local Mistral Small 3.2 through Ollama's
OpenAI-compatible endpoint (no API key, no data leaves the machine).

Mapping to the SDK:
  input guardrail   guards.scope_violation: out-of-scope requests trip before the model is called
  function tools    the two solvers, with the JSON schemas of physics_ai_demo.TOOLS; every call passes the
                    grounding check, the reused-quantity check and the solver's own validator before any number
                    is computed
  stop_on_first_tool  the tool result is the answer; a second, tool-less agent only explains it

Returns the same record shape as physics_ai_demo.run_request, so eval.py scores both pipelines identically.
Usage: .venv/bin/python agents_sdk_pipeline.py "request"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from agents import (Agent, FunctionTool, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, ModelSettings,
                    OpenAIChatCompletionsModel, RunContextWrapper, Runner, input_guardrail, set_tracing_disabled)
from openai import AsyncOpenAI

from guards import reused_quantity, scope_violation
from physics_ai_demo import (EXPLAIN_PROMPT, OLLAMA_MODEL, SOLVERS, SYSTEM_PROMPT, TOOLS, InputRejected,
                             ungrounded_arguments)

set_tracing_disabled(True)
CLIENT = AsyncOpenAI(base_url=os.environ.get("OLLAMA_V1", "http://localhost:11434/v1"), api_key="ollama")
MODEL = OpenAIChatCompletionsModel(model=OLLAMA_MODEL, openai_client=CLIENT)
SETTINGS = ModelSettings(temperature=0, seed=0) if "seed" in ModelSettings.__dataclass_fields__ else ModelSettings(temperature=0)


@input_guardrail
async def scope_guard(ctx: RunContextWrapper[dict], agent: Agent, user_input: Any) -> GuardrailFunctionOutput:
    hit = scope_violation(user_input if isinstance(user_input, str) else json.dumps(user_input))
    return GuardrailFunctionOutput(output_info=hit, tripwire_triggered=hit is not None)


def make_tool(spec: dict[str, Any]) -> FunctionTool:
    name = spec["function"]["name"]

    async def invoke(ctx: RunContextWrapper[dict], raw: str) -> str:
        state = ctx.context
        try:
            arguments = json.loads(raw or "{}")
        except json.JSONDecodeError:
            state["record"] = {"status": "rejected", "reason": "The model returned invalid tool arguments."}
            return json.dumps(state["record"])
        state["record"] = {"tool_call": {"name": name, "arguments": arguments}}
        query = state["query"]
        bad = ungrounded_arguments(query, arguments)
        reused = reused_quantity(query, arguments)
        if bad or reused:
            why = []
            if bad:
                why.append(f"{', '.join(bad)} do not match any value stated in the request")
            if reused:
                why.append(f"{', '.join(reused)} were filled from one single stated number")
            state["record"].update(status="rejected", reason="Not computed: " + "; ".join(why) + ". Please state these values explicitly.")
            return json.dumps(state["record"])
        try:
            result = SOLVERS[name](arguments)
        except InputRejected as exc:
            state["record"].update(status="rejected", reason=str(exc))
            return json.dumps(state["record"])
        state["record"].update(result)
        return json.dumps(result)

    return FunctionTool(name=name, description=spec["function"]["description"],
                        params_json_schema=spec["function"]["parameters"], on_invoke_tool=invoke,
                        strict_json_schema=False)


INTERPRETER = Agent(name="interpreter", instructions=SYSTEM_PROMPT, model=MODEL, model_settings=SETTINGS,
                    tools=[make_tool(t) for t in TOOLS], input_guardrails=[scope_guard],
                    tool_use_behavior="stop_on_first_tool")
EXPLAINER = Agent(name="explainer", instructions=EXPLAIN_PROMPT, model=MODEL, model_settings=SETTINGS)


async def arun(query: str, explain: bool = True) -> dict[str, Any]:
    state: dict[str, Any] = {"query": query}
    record: dict[str, Any] = {"backend": "ollama+agents-sdk", "model_id": OLLAMA_MODEL,
                              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        out = await Runner.run(INTERPRETER, query, context=state, max_turns=2)
    except InputGuardrailTripwireTriggered as exc:
        hit = exc.guardrail_result.output.output_info
        return {**record, "status": "rejected", "guard": "scope",
                "reason": f"Out of scope for these steady, simply supported tools ('{hit}'). Not computed."}
    if "record" not in state:
        return {**record, "status": "needs_clarification", "message": str(out.final_output)}
    record.update(state["record"])
    if record.get("status") == "calculated" and explain:
        prompt = f"Request: {query}\nTool result: {json.dumps({k: record[k] for k in record if k not in ('backend', 'created_at_utc')})}"
        record["explanation"] = str((await Runner.run(EXPLAINER, prompt)).final_output)
    return record


LOOP = asyncio.new_event_loop()  # one loop for all requests: the HTTP client is bound to the loop it first ran on


def run_request(query: str, backend: str = "ollama", explain: bool = True, grounding: bool = True) -> dict[str, Any]:
    return LOOP.run_until_complete(arun(query, explain))


if __name__ == "__main__":
    print(json.dumps(run_request(sys.argv[1]), indent=2, ensure_ascii=False))
