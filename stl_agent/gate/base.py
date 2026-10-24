"""Gate types and the decision rules, kept separate from any model call so they can be tested directly."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

MAX_ROUNDS = int(os.environ.get("CAD_GATE_MAX_ROUNDS", "2"))
MAX_QUESTIONS = 3

PROCEED, CLARIFY, OUT_OF_SCOPE = "proceed", "clarify", "out_of_scope"


@dataclass
class Decision:
    action: str  # PROCEED | CLARIFY | OUT_OF_SCOPE
    spec: str  # self-contained part description handed to generate_cad
    scope: str = "in_scope"  # in_scope | simplified | out_of_scope
    scope_reason: str = ""
    questions: list[str] = field(default_factory=list)
    assumptions: list[dict] = field(default_factory=list)  # [{"param", "default", "critical", ...}]
    note: str = ""  # e.g. why the gate fell back to proceeding
    name: str = ""  # short file-name slug for the part, e.g. "flange-80mm-6-holes"
    # Gate-picked library examples (CAD_GATE_PICKS): None = the gate wasn't asked to pick,
    # [] = it found nothing relevant.
    example_ids: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Gate(Protocol):
    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision: ...



def slugify(text: str, max_words: int = 6, max_len: int = 48) -> str:
    """'An 80mm Flange, 6 holes!' -> 'an-80mm-flange-6-holes'. Safe as a file name; may be empty."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words)[:max_len].strip("-")



def decide_from_report(report: dict, fallback_spec: str, rounds_asked: int, max_rounds: int = MAX_ROUNDS) -> Decision:
    """The decision rules. Kept separate from the LLM call so they can be tested directly."""
    scope = str(report.get("scope", "in_scope")).strip().lower()
    if scope not in ("in_scope", "simplified", "out_of_scope"):
        scope = "in_scope"
    spec = str(report.get("spec") or "").strip() or fallback_spec
    unknowns = [u for u in report.get("unknowns") or [] if isinstance(u, dict)]
    critical = [u for u in unknowns if u.get("critical") is True]
    d = Decision(action=PROCEED, spec=spec, scope=scope, scope_reason=str(report.get("scope_reason") or ""),
                 assumptions=unknowns, name=slugify(str(report.get("name") or "")))
    if "examples" in report:
        picks = report["examples"] if isinstance(report["examples"], list) else []
        d.example_ids = list(dict.fromkeys(str(x).strip() for x in picks if str(x).strip()))

    if scope == "out_of_scope":
        d.action = OUT_OF_SCOPE
    elif critical and rounds_asked < max_rounds:
        d.action = CLARIFY
        d.questions = [str(u.get("question") or f"What should {u.get('param')} be?") for u in critical][:MAX_QUESTIONS]
    elif critical:
        d.note = f"Still missing critical details after {rounds_asked} round(s); proceeding with best guesses."
    return d



class PassthroughGate:
    """CAD_GATE=off: the old behaviour, every request goes straight to generation."""

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        return Decision(action=PROCEED, spec="\n".join(t["content"] for t in conversation if t["role"] == "user"))


