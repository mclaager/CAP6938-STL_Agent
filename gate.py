"""Intake gate: decides whether a request is ready to build, needs clarifying questions,
or is out of scope, before any CadQuery generation is attempted.

The model only reports facts about the request (scope, filled-in spec, unknown parameters
and whether each one is critical). The decision itself is made by fixed rules in `decide`,
so it is predictable and easy to evaluate.

Config via environment variables:
    CAD_GATE              "llm" (default), "off" (always proceed), or "jev" (not implemented yet)
    CAD_GATE_MAX_ROUNDS   clarifying rounds before proceeding on assumptions (default 2)

Gates implement `decide(conversation, rounds_asked) -> Decision`, where `conversation` is a list
of {"role": "user" | "assistant", "content": str} turns from the intake chat.
"""

from __future__ import annotations

import json
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

    def to_dict(self) -> dict:
        return asdict(self)


class Gate(Protocol):
    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision: ...


GATE_PROMPT = """\
You are the intake step of a text-to-CAD agent. A later step writes CadQuery (Python parametric CAD)
code for the part, so it can build anything made from extrude, revolve, loft, sweep, boolean
union/cut, holes, fillets, chamfers, shells, patterns and simple text. Units are millimetres.

Read the conversation and reply with ONE JSON object in a ```json block, nothing else:
{
  "scope": "in_scope" | "simplified" | "out_of_scope",
  "scope_reason": "one short sentence; empty when in_scope",
  "spec": "complete, self-contained description of the part to build, with every dimension in mm, including everything the user said in later messages",
  "name": "short file name for the part: 2-5 lowercase words joined by hyphens, e.g. flange-80mm-6-holes",
  "unknowns": [
    {"param": "name", "default": "value you would use", "critical": true | false,
     "question": "short question to ask the user about it"}
  ]
}

SCOPE
- in_scope: mechanical or geometric parts: plates, brackets, enclosures, flanges, spacers, gears,
  knobs, hooks, mounts, containers, organizers, simple text plaques.
- simplified: only buildable as a stylized geometric approximation (a low-poly vase, a simple
  rocket, a chess pawn). The spec must describe that simplified version.
- out_of_scope: organic or sculpted shapes (people, animals, faces, characters, realistic plants),
  copies of images or logos, or anything that is not a single physical part.

UNKNOWNS
List every detail the user did not give that you had to decide. Mark critical=true ONLY when a
wrong guess would likely make the part useless:
- it must fit or attach to a specific real object whose dimensions are not given and not standard
  (e.g. "a case for my phone", "a cap for this bottle");
- the guess changes the basic shape (how many compartments, open vs closed, which way it mounts).
NOT critical: overall size of a generic part, wall thickness, fillet radii, hole spacing, text font.
Known standards are not unknowns: use their real dimensions (M3/M4 screws, NEMA 17 motors, 608
bearings, AA batteries, GoPro mounts, 2020 extrusion).
If the user already answered something, or said "you choose" / "anything", it is not critical.
"""


def _transcript(conversation: list[dict]) -> str:
    lines = []
    for turn in conversation:
        who = "User" if turn["role"] == "user" else "Agent"
        lines.append(f"{who}: {turn['content'].strip()}")
    return "Conversation so far:\n\n" + "\n\n".join(lines)


def slugify(text: str, max_words: int = 6, max_len: int = 48) -> str:
    """'An 80mm Flange, 6 holes!' -> 'an-80mm-flange-6-holes'. Safe as a file name; may be empty."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words)[:max_len].strip("-")


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S | re.I)


def parse_gate_reply(text: str) -> dict:
    """Pull the JSON object out of the reply (fenced block preferred, else the outermost braces)."""
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("gate reply is not a JSON object")
    return data


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

    if scope == "out_of_scope":
        d.action = OUT_OF_SCOPE
    elif critical and rounds_asked < max_rounds:
        d.action = CLARIFY
        d.questions = [str(u.get("question") or f"What should {u.get('param')} be?") for u in critical][:MAX_QUESTIONS]
    elif critical:
        d.note = f"Still missing critical details after {rounds_asked} round(s); proceeding with best guesses."
    return d


class LLMGate:
    def __init__(self, llm, max_rounds: int = MAX_ROUNDS):
        self.llm, self.max_rounds = llm, max_rounds

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        fallback_spec = "\n".join(t["content"] for t in conversation if t["role"] == "user")
        reply = self.llm.complete(GATE_PROMPT, [{"role": "user", "content": _transcript(conversation)}])
        try:
            report = parse_gate_reply(reply)
        except (ValueError, json.JSONDecodeError):
            # Fail open: a broken gate reply should not block generation.
            return Decision(action=PROCEED, spec=fallback_spec, note="Gate reply was not valid JSON; using the raw request.")
        return decide_from_report(report, fallback_spec, rounds_asked, self.max_rounds)


class PassthroughGate:
    """CAD_GATE=off: the old behaviour, every request goes straight to generation."""

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        return Decision(action=PROCEED, spec="\n".join(t["content"] for t in conversation if t["role"] == "user"))


class JevGate:
    """Placeholder for a TypeSafe Jev backend (calibrated classifier, limited early access).

    Plan: send the transcript as `state` with a `choice` question for scope and `noul` questions
    such as "Must the part fit an object whose dimensions are not given?", then gate on the returned
    confidence. Jev cannot write questions or a spec, so CLARIFY still needs an LLM call.
    """

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        raise NotImplementedError("Jev gate is not implemented yet; set CAD_GATE=llm or off.")


def make_gate(llm, kind: str | None = None) -> Gate:
    kind = (kind or os.environ.get("CAD_GATE", "llm")).lower()
    if kind == "llm":
        return LLMGate(llm)
    if kind == "off":
        return PassthroughGate()
    if kind == "jev":
        return JevGate()
    raise ValueError(f"Unknown CAD_GATE {kind!r}; use 'llm', 'off' or 'jev'.")


if __name__ == "__main__":
    # Offline check of the decision rules.
    base = {"scope": "in_scope", "spec": "phone stand", "unknowns": [
        {"param": "phone width", "default": "75 mm", "critical": True, "question": "How wide is your phone?"},
        {"param": "angle", "default": "60 deg", "critical": False, "question": "What angle?"},
    ]}
    assert decide_from_report(base, "x", 0).action == CLARIFY
    assert decide_from_report(base, "x", 0).questions == ["How wide is your phone?"]
    assert decide_from_report(base, "x", MAX_ROUNDS).action == PROCEED
    assert decide_from_report({**base, "unknowns": base["unknowns"][1:]}, "x", 0).action == PROCEED
    assert decide_from_report({"scope": "out_of_scope"}, "x", 0).action == OUT_OF_SCOPE
    assert decide_from_report({}, "raw", 0).spec == "raw"
    assert decide_from_report({**base, "name": "Phone Stand/../x"}, "x", 0).name == "phone-stand-x"
    assert slugify("An 80mm Flange, 6 holes!") == "an-80mm-flange-6-holes"
    assert parse_gate_reply('ok\n```json\n{"scope": "in_scope"}\n```')["scope"] == "in_scope"
    print("gate rule checks passed")
