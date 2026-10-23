"""Intake gate: decides whether a request is ready to build, needs clarifying questions,
or is out of scope, before any CadQuery generation is attempted.

The model only reports facts about the request (scope, filled-in spec, unknown parameters
and whether each one is critical). The decision itself is made by fixed rules in
`base.decide_from_report`, so it is predictable and easy to evaluate.

Modules: base.py (types + rules), llm_gate.py (LLM-backed gate), jev.py (Jev stub, issue #12).

Config via environment variables:
    CAD_GATE              "llm" (default), "off" (always proceed), or "jev" (not implemented yet)
    CAD_GATE_MAX_ROUNDS   clarifying rounds before proceeding on assumptions (default 2)

Gates implement `decide(conversation, rounds_asked) -> Decision`, where `conversation` is a list
of {"role": "user" | "assistant", "content": str} turns from the intake chat.
"""

from __future__ import annotations

import os

from .base import (CLARIFY, MAX_QUESTIONS, MAX_ROUNDS, OUT_OF_SCOPE, PROCEED, Decision, Gate,
                   PassthroughGate, decide_from_report, slugify)
from .jev import JevGate
from .llm_gate import GATE_PROMPT, LLMGate, parse_gate_reply

def make_gate(llm, kind: str | None = None) -> Gate:
    kind = (kind or os.environ.get("CAD_GATE", "llm")).lower()
    if kind == "llm":
        return LLMGate(llm)
    if kind == "off":
        return PassthroughGate()
    if kind == "jev":
        return JevGate()
    raise ValueError(f"Unknown CAD_GATE {kind!r}; use 'llm', 'off' or 'jev'.")
