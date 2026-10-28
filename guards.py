"""Deterministic guards added in v2, on top of the grounding check in physics_ai_demo.py.

scope_violation   the request describes something the two tools do not model (transient loads, other supports
                  or load cases, 2-D/3-D geometry, radiation), even when every number is stated (eval case 21).
reused_quantity   two arguments were filled from the same single number in the request, e.g. the stated depth
                  also used as the width (eval case 16). A number may serve two arguments only if it is written
                  twice or the text says the section is square.
"""

from __future__ import annotations

import math
import re
from typing import Any

from physics_ai_demo import BARE_FACTORS, SUFFIX_DIMENSION, stated_quantities

SCOPE = re.compile(
    r"\bpulse|transient|time[- ]dependent|\bramp|\bafter\s+\d|\d+(\.\d+)?\s*(ms|s|sec|seconds?|min|minutes?)\b(?![/²\w])"
    r"|monoblock|cooling tube|\btube\b|cantilever|distributed|fixed[- ]end|clamped|built[- ]in|radiat"
    r"|temperature[- ]dependent",
    re.I,
)


def scope_violation(query: str) -> str | None:
    m = SCOPE.search(query)
    return m.group(0) if m else None


def _source_indices(query: str, name: str, value: float) -> set[int]:
    """Indices of the stated quantities this argument value could have come from."""
    dimension = next((d for suffix, d in SUFFIX_DIMENSION if name.endswith(suffix)), None)
    hits = set()
    for i, (d, v) in enumerate(stated_quantities(query)):
        factors = (1.0,) if d == dimension else BARE_FACTORS.get(dimension, (1.0,)) if d == "bare" else ()
        if any(math.isclose(value, v * f, rel_tol=5e-3) for f in factors):
            hits.add(i)
    return hits


def reused_quantity(query: str, arguments: dict[str, Any]) -> list[str]:
    if re.search(r"\bsquare\b", query, re.I):
        return []
    numeric = {k: v for k, v in arguments.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    sources = {k: _source_indices(query, k, v) for k, v in numeric.items()}
    bad = []
    names = list(sources)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Both arguments can only have come from one and the same written number.
            if len(sources[a]) == 1 and sources[a] == sources[b]:
                bad += [a, b]
    return sorted(set(bad))
