"""Intake gate: decides whether a request is ready to build, needs clarifying questions,
or is out of scope, before any CadQuery generation is attempted.

The model only reports facts about the request (scope, filled-in spec, unknown parameters
and whether each one is critical). The decision itself is made by fixed rules in
`base.decide_from_report`, so it is predictable and easy to evaluate.

Modules: base.py (types + rules), llm_gate.py (LLM-backed gate), jev.py (Jev stub, issue #12).

Settings (settings.toml, see config.py):
    gate              "llm" (default), "off" (always proceed), or "jev" (not implemented yet)
    gate_max_rounds   clarifying rounds before proceeding on assumptions (default 2)
    gate_picks        let the gate pick the RAG examples

Gates implement `decide(conversation, rounds_asked) -> Decision`, where `conversation` is a list
of {"role": "user" | "assistant", "content": str} turns from the intake chat.
"""

from __future__ import annotations

from ..config import settings
from .base import (CLARIFY, MAX_QUESTIONS, MAX_ROUNDS, OUT_OF_SCOPE, PROCEED, Decision, Gate,
                   PassthroughGate, decide_from_report, slugify)
from .jev import JevGate
from .llm_gate import GATE_PROMPT, LLMGate, parse_gate_reply

def make_gate(llm, kind: str | None = None, library: list[dict] | None = None) -> Gate:
    """`library`: example dicts the LLM gate may pick from (gate_picks setting); None to not pick."""
    kind = (kind or settings().gate).lower()
    if kind == "llm":
        return LLMGate(llm, library=library)
    if kind == "off":
        return PassthroughGate()
    if kind == "jev":
        return JevGate()
    raise ValueError(f"Unknown gate {kind!r}; use 'llm', 'off' or 'jev'.")
