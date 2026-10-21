#!/usr/bin/env python3
"""Mistral tool-calling demo: the model interprets the request, deterministic physics computes the answer.

Two backends run the same prompt and tools:
  --backend ollama   local open-weight Mistral Small 3.2 through Ollama (default, no key needed)
  --backend mistral  Mistral La Plateforme (needs MISTRAL_API_KEY and the mistralai package)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = "mistral-small3.2"
MISTRAL_MODEL = "mistral-small-latest"
BEAM_TOOL = "calculate_center_loaded_beam"
HEAT_TOOL = "calculate_layered_heat_flux"
W_MELTING_C = 3422.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": BEAM_TOOL,
            "description": (
                "Calculate elastic bending response for a simply supported, "
                "prismatic rectangular beam with one centered point load. "
                "Use SI units: meters, newtons, and pascals. Do not use for "
                "other geometries, load positions, supports, or nonlinear cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "length_m": {"type": "number", "description": "Beam span in meters."},
                    "width_m": {"type": "number", "description": "Rectangular cross-section width in meters."},
                    "height_m": {"type": "number", "description": "Rectangular cross-section depth in meters, in the bending direction."},
                    "load_n": {"type": "number", "description": "Centered downward point load in newtons."},
                    "elastic_modulus_pa": {"type": "number", "description": "Young's modulus in pascals."},
                    "material": {"type": "string", "description": "Material name supplied or explicitly stated by the user."},
                    "support": {"type": "string", "enum": ["simply_supported"]},
                    "load_case": {"type": "string", "enum": ["centered_point_load"]},
                },
                "required": [
                    "length_m", "width_m", "height_m", "load_n",
                    "elastic_modulus_pa", "material", "support", "load_case"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": HEAT_TOOL,
            "description": (
                "Steady-state, one-dimensional temperatures through a tungsten armour layer bonded to a "
                "copper heat sink that is cooled by water on its back face, under a uniform surface heat flux. "
                "Use SI units: W/m^2, meters, W/(m K), W/(m^2 K), degrees Celsius. Do not use for transient "
                "loads, two- or three-dimensional geometry (for example monoblocks with a cooling tube), "
                "radiation, or temperature-dependent conductivity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "heat_flux_w_m2": {"type": "number", "description": "Uniform heat flux on the tungsten surface in W/m^2."},
                    "tungsten_thickness_m": {"type": "number", "description": "Tungsten layer thickness in meters."},
                    "tungsten_conductivity_w_mk": {"type": "number", "description": "Tungsten thermal conductivity in W/(m K)."},
                    "copper_thickness_m": {"type": "number", "description": "Copper layer thickness between the bond and the coolant, in meters."},
                    "copper_conductivity_w_mk": {"type": "number", "description": "Copper thermal conductivity in W/(m K)."},
                    "coolant_htc_w_m2k": {"type": "number", "description": "Heat transfer coefficient to the coolant in W/(m^2 K)."},
                    "coolant_temperature_c": {"type": "number", "description": "Bulk coolant temperature in degrees Celsius."},
                },
                "required": [
                    "heat_flux_w_m2", "tungsten_thickness_m", "tungsten_conductivity_w_mk",
                    "copper_thickness_m", "copper_conductivity_w_mk", "coolant_htc_w_m2k",
                    "coolant_temperature_c",
                ],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the request interpreter for two constrained physics tools:
a simply supported beam with a centered point load, and steady one-dimensional heat
conduction through a tungsten layer on a water-cooled copper heat sink.
Call a tool only when the request matches its scope exactly and states every required
value. Convert supplied units to SI. Never guess or fill in missing values, material
properties, supports, load positions, or boundary conditions, even if the user asks you
to assume typical values. If the request is unsupported or underspecified, do not call a
tool: say what is missing or out of scope in one or two sentences. Do not describe the
calculation as finite element analysis or as a design approval. The Python tools are the
sole source of numerical results."""

EXPLAIN_PROMPT = """Explain the tool result to an engineer in at most four sentences.
Quote numbers exactly as the tool returned them, with their units, and state the main
limitation. Do not introduce any number that is not in the tool result."""


class InputRejected(ValueError):
    pass


def _numbers(raw: dict[str, Any], names: tuple[str, ...], allow_zero: tuple[str, ...] = ()) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputRejected(f"{name} must be a number in SI units.")
        if not math.isfinite(value) or value < 0 or (value == 0 and name not in allow_zero):
            raise InputRejected(f"{name} must be finite and greater than zero.")
        values[name] = float(value)
    return values


def _exact_fields(raw: dict[str, Any], expected: set[str]) -> None:
    if set(raw) != expected:
        raise InputRejected("The tool request must contain exactly the documented fields.")


@dataclass(frozen=True)
class BeamCase:
    length_m: float
    width_m: float
    height_m: float
    load_n: float
    elastic_modulus_pa: float
    material: str
    support: str
    load_case: str


def calculate_beam(raw: dict[str, Any]) -> dict[str, Any]:
    _exact_fields(raw, set(BeamCase.__dataclass_fields__))
    values = _numbers(raw, ("length_m", "width_m", "height_m", "load_n", "elastic_modulus_pa"))
    if values["length_m"] > 100 or values["width_m"] > 10 or values["height_m"] > 10:
        raise InputRejected("Dimensions exceed the supported screening-model range.")
    if values["load_n"] > 1e9 or not 1e8 <= values["elastic_modulus_pa"] <= 1.5e12:
        raise InputRejected("Load or elastic modulus is outside the supported range (E between 0.1 GPa and 1500 GPa).")
    material = raw["material"]
    if not isinstance(material, str) or not material.strip():
        raise InputRejected("material must be a non-empty string.")
    if raw["support"] != "simply_supported":
        raise InputRejected("Only simply_supported beams are supported.")
    if raw["load_case"] != "centered_point_load":
        raise InputRejected("Only centered_point_load is supported.")
    case = BeamCase(**values, material=material.strip(), support="simply_supported", load_case="centered_point_load")

    second_moment_m4 = case.width_m * case.height_m**3 / 12.0
    maximum_moment_nm = case.load_n * case.length_m / 4.0
    maximum_stress_pa = maximum_moment_nm * (case.height_m / 2.0) / second_moment_m4
    maximum_deflection_m = case.load_n * case.length_m**3 / (48.0 * case.elastic_modulus_pa * second_moment_m4)
    return {
        "status": "calculated",
        "tool": BEAM_TOOL,
        "model": "Euler-Bernoulli simply supported beam, centered point load",
        "inputs_si": asdict(case),
        "equations": {
            "second_moment_m4": "I = b h^3 / 12",
            "maximum_bending_stress_pa": "sigma_max = (P L / 4) (h / 2) / I",
            "maximum_deflection_m": "delta_max = P L^3 / (48 E I)",
        },
        "results": {
            "maximum_bending_stress_mpa": round(maximum_stress_pa / 1e6, 4),
            "maximum_deflection_mm": round(maximum_deflection_m * 1e3, 4),
        },
        "limitations": [
            "Linear elastic material response and small deflection are assumed.",
            "Shear deformation, thermal strain, plasticity, contact, residual stress, and local stress concentrations are excluded.",
            "This screening calculation is not a finite element result or engineering release.",
        ],
    }


HEAT_FIELDS = (
    "heat_flux_w_m2", "tungsten_thickness_m", "tungsten_conductivity_w_mk",
    "copper_thickness_m", "copper_conductivity_w_mk", "coolant_htc_w_m2k",
)


def calculate_heat(raw: dict[str, Any]) -> dict[str, Any]:
    _exact_fields(raw, set(HEAT_FIELDS) | {"coolant_temperature_c"})
    v = _numbers(raw, HEAT_FIELDS)
    coolant = raw["coolant_temperature_c"]
    if isinstance(coolant, bool) or not isinstance(coolant, (int, float)) or not math.isfinite(coolant):
        raise InputRejected("coolant_temperature_c must be a number in degrees Celsius.")
    if not 0 <= coolant <= 400:
        raise InputRejected("Coolant temperature must be between 0 and 400 degrees Celsius for a water coolant.")
    if v["heat_flux_w_m2"] > 1e8:
        raise InputRejected("Heat flux above 100 MW/m^2 is outside the supported range.")
    if v["tungsten_thickness_m"] > 0.05 or v["copper_thickness_m"] > 0.05:
        raise InputRejected("Layer thicknesses above 50 mm are outside the supported range.")
    for name in ("tungsten_conductivity_w_mk", "copper_conductivity_w_mk"):
        if not 1 <= v[name] <= 1000:
            raise InputRejected(f"{name} must be between 1 and 1000 W/(m K).")
    if not 100 <= v["coolant_htc_w_m2k"] <= 1e6:
        raise InputRejected("Heat transfer coefficient must be between 100 and 1e6 W/(m^2 K).")

    q = v["heat_flux_w_m2"]
    t_wall = coolant + q / v["coolant_htc_w_m2k"]
    t_bond = t_wall + q * v["copper_thickness_m"] / v["copper_conductivity_w_mk"]
    t_surface = t_bond + q * v["tungsten_thickness_m"] / v["tungsten_conductivity_w_mk"]
    warnings = []
    if t_surface >= W_MELTING_C:
        warnings.append(f"Computed tungsten surface temperature is at or above the tungsten melting point ({W_MELTING_C:.0f} C); the steady state is not physical.")
    return {
        "status": "calculated",
        "tool": HEAT_TOOL,
        "model": "Steady 1-D conduction, W on Cu, convective cooling, constant conductivities",
        "inputs_si": {**v, "coolant_temperature_c": float(coolant)},
        "equations": {
            "cooled_wall_c": "T_wall = T_coolant + q / h",
            "w_cu_bond_c": "T_bond = T_wall + q t_Cu / k_Cu",
            "w_surface_c": "T_surface = T_bond + q t_W / k_W",
        },
        "results": {
            "w_surface_temperature_c": round(t_surface, 1),
            "w_cu_bond_temperature_c": round(t_bond, 1),
            "cooled_wall_temperature_c": round(t_wall, 1),
        },
        "warnings": warnings,
        "limitations": [
            "One-dimensional steady state: no monoblock or tube geometry, no transients, no radiation.",
            "Constant conductivities and a perfect W/Cu bond (no contact resistance) are assumed.",
            "This screening calculation is not a finite element result or engineering release.",
        ],
    }


SOLVERS = {BEAM_TOOL: calculate_beam, HEAT_TOOL: calculate_heat}

# Grounding check: every numeric argument must equal a quantity the user wrote, converted with the unit the user
# wrote next to it (parsed here, not by the model), and every categorical argument must be stated in the request.
# This catches invented values and unit-conversion slips before anything is computed.
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")
SQ = r"(?:²|2|\^2)"
UNITS = [  # (pattern right after the number, dimension, factor to SI); first match wins
    (rf"kW\s*/\s*\(?m{SQ}\s*(?:·|\*|/)?\s*K\)?", "htc", 1e3),
    (rf"W\s*/\s*\(?m{SQ}\s*(?:·|\*|/)?\s*K\)?", "htc", 1.0),
    (r"W\s*/\s*\(?m\s*(?:·|\*|/)?\s*K\)?", "conductivity", 1.0),
    (rf"MW\s*(?:/\s*m{SQ}|per square met(?:re|er))", "heat_flux", 1e6),
    (rf"kW\s*(?:/\s*m{SQ}|per square met(?:re|er))", "heat_flux", 1e3),
    (rf"W\s*(?:/\s*m{SQ}|per square met(?:re|er))", "heat_flux", 1.0),
    (r"GPa", "pressure", 1e9), (r"MPa", "pressure", 1e6), (r"kPa", "pressure", 1e3), (r"Pa\b", "pressure", 1.0),
    (r"kN\b", "force", 1e3), (r"MN\b", "force", 1e6), (r"N\b", "force", 1.0),
    (r"mm\b", "length", 1e-3), (r"cm\b", "length", 1e-2), (r"m\b", "length", 1.0),
    (r"°\s*C|C\b", "temperature", 1.0),
]
# Bare numbers (no unit after them, as in "20 x 40 mm") may carry the usual unit prefixes of the argument's dimension.
BARE_FACTORS = {"length": (1.0, 1e-3, 1e-2), "force": (1.0, 1e3), "pressure": (1.0, 1e6, 1e9), "heat_flux": (1.0, 1e6),
                "conductivity": (1.0,), "htc": (1.0, 1e3), "temperature": (1.0,)}
SUFFIX_DIMENSION = [("_w_m2k", "htc"), ("_w_mk", "conductivity"), ("_w_m2", "heat_flux"), ("_pa", "pressure"),
                    ("_n", "force"), ("_c", "temperature"), ("_m", "length")]
STATED = {
    "simply_supported": re.compile(r"simply[- ]supported|two supports|between (two )?supports", re.I),
    "centered_point_load": re.compile(r"cent(er|re)|middle|mid-?span|midspan|cent(er|re)d", re.I),
}


def stated_quantities(text: str) -> list[tuple[str, float]]:
    """(dimension, SI value) for every number in the text; dimension 'bare' when no unit follows."""
    found = []
    for m in NUMBER.finditer(text):
        value = float(m.group().replace(",", ""))
        rest = text[m.end():m.end() + 30].lstrip()
        for pattern, dimension, factor in UNITS:
            if re.match(pattern, rest):
                found.append((dimension, value * factor))
                break
        else:
            found.append(("bare", value))
    return found


def ungrounded_arguments(query: str, arguments: dict[str, Any]) -> list[str]:
    stated = stated_quantities(query)
    bad = []
    for name, value in arguments.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            dimension = next((d for suffix, d in SUFFIX_DIMENSION if name.endswith(suffix)), None)
            candidates = [v for d, v in stated if d == dimension]
            candidates += [v * f for d, v in stated if d == "bare" for f in BARE_FACTORS.get(dimension, (1.0,))]
            if not any(math.isclose(value, c, rel_tol=5e-3) for c in candidates):
                bad.append(name)
        elif value in STATED and not STATED[value].search(query):
            bad.append(name)
    return bad


def chat(backend: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """One chat turn. Returns {'content': str, 'tool_calls': [{'id', 'name', 'arguments': dict|str}]}."""
    if backend == "ollama":
        body = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0, "seed": 0}}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        msg = json.load(urllib.request.urlopen(req, timeout=600))["message"]
        calls = [
            {"id": f"call_{i}", "name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        return {"content": msg.get("content") or "", "tool_calls": calls}

    from mistralai import Mistral  # imported lazily so the local backend needs no extra package

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    kwargs: dict[str, Any] = {"model": MISTRAL_MODEL, "messages": messages, "temperature": 0}
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    msg = client.chat.complete(**kwargs).choices[0].message
    calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in (msg.tool_calls or [])]
    return {"content": msg.content or "", "tool_calls": calls}


def run_request(query: str, backend: str = "ollama", explain: bool = True, grounding: bool = True) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    first = chat(backend, messages, TOOLS)
    record: dict[str, Any] = {
        "backend": backend,
        "model_id": OLLAMA_MODEL if backend == "ollama" else MISTRAL_MODEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not first["tool_calls"]:
        return {**record, "status": "needs_clarification", "message": first["content"]}
    if len(first["tool_calls"]) != 1:
        return {**record, "status": "rejected", "reason": "The model requested more than one tool call."}

    call = first["tool_calls"][0]
    record["tool_call"] = {"name": call["name"], "arguments": call["arguments"]}
    if call["name"] not in SOLVERS:
        return {**record, "status": "rejected", "reason": f"Unknown tool {call['name']!r}."}
    arguments = call["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {**record, "status": "rejected", "reason": "The model returned invalid tool arguments."}
    if grounding and isinstance(arguments, dict):
        bad = ungrounded_arguments(query, arguments)
        if bad:
            return {**record, "status": "rejected",
                    "reason": f"Not computed: {', '.join(bad)} do not match any value stated in the request "
                              "(checked with the units written there). Please state these values explicitly."}
    try:
        result = SOLVERS[call["name"]](arguments)
    except InputRejected as exc:
        return {**record, "status": "rejected", "reason": str(exc)}

    record.update(result)
    if explain:
        messages += [
            {"role": "assistant", "content": first["content"], "tool_calls": [
                # Ollama expects the arguments as an object, the Mistral API as a JSON string.
                {"id": call["id"], "type": "function", "function": {
                    "name": call["name"], "arguments": arguments if backend == "ollama" else json.dumps(arguments)}}]},
            {"role": "tool", "name": call["name"], "tool_call_id": call["id"], "content": json.dumps(result)},
            {"role": "user", "content": EXPLAIN_PROMPT},
        ]
        record["explanation"] = chat(backend, messages, None)["content"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", help="Natural-language engineering request")
    parser.add_argument("--backend", choices=("ollama", "mistral"), default="ollama")
    parser.add_argument("--output", help="Optional path for a JSON result record")
    parser.add_argument("--no-grounding", action="store_true", help="Skip the check that every argument was stated by the user")
    args = parser.parse_args()
    if args.backend == "mistral" and not os.environ.get("MISTRAL_API_KEY"):
        parser.error("Set MISTRAL_API_KEY in the environment. The key is never saved to the result.")

    result = run_request(args.request, args.backend, grounding=not args.no_grounding)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
    return 0 if result["status"] in ("calculated", "needs_clarification") else 2


if __name__ == "__main__":
    sys.exit(main())
