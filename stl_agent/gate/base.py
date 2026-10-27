"""Gate types and the decision rules, kept separate from any model call so they can be tested directly."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..config import settings

MAX_ROUNDS = 2  # the usual value; the setting is gate_max_rounds (settings.defaults.toml)
MAX_QUESTIONS = 3

PROCEED, CLARIFY, OUT_OF_SCOPE = "proceed", "clarify", "out_of_scope"
SCOPES = ("in_scope", "simplified", "out_of_scope", "composite")

# Combined parts (#10): the relations and alignments sources.assembly can apply.
RELATIONS = ("on_top", "under", "beside", "inside")
SIDES = ("left", "right", "front", "back")
MAX_PIECES = 5
MAX_SEARCHES = 3


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
    search: str = ""  # a few words to find an existing model of it on a model site, e.g. "horse statue"
    # scope "composite": the pieces to make and join, checked by parse_pieces (first one is the base).
    pieces: list[dict] = field(default_factory=list)
    # Gate-picked library examples (gate_picks setting): None = the gate wasn't asked to pick,
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



def _size(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 10_000 else None


def parse_pieces(raw) -> tuple[list[dict], list[str]]:
    """The gate's piece plan, checked: (valid pieces, notes on what was dropped). Each piece comes out as
    {"id", "source": "cadquery" | "thingiverse", "spec" | "search": [...] + "fallback_spec", "height_mm",
    "place": {"on", "relation", "side", "align"}}; the first piece has no "place"."""
    pieces, notes, seen = [], [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        pid = slugify(str(item.get("id") or ""), max_words=4, max_len=24)
        if not pid or pid in seen:
            notes.append(f"Dropped a piece with a missing or repeated id ({item.get('id')!r}).")
            continue
        source = str(item.get("source") or "").strip().lower()
        p = {"id": pid, "source": source, "height_mm": _size(item.get("height_mm"))}
        if source == "cadquery":
            p["spec"] = str(item.get("spec") or "").strip()
            if not p["spec"]:
                notes.append(f"Dropped {pid}: no spec to build it from.")
                continue
        elif source == "thingiverse":
            search = item.get("search") or pid.replace("-", " ")
            search = [search] if isinstance(search, str) else search if isinstance(search, list) else []
            p["search"] = [" ".join(str(s).split()) for s in search if str(s).strip()][:MAX_SEARCHES]
            p["fallback_spec"] = str(item.get("fallback_spec") or "").strip()
        else:
            notes.append(f"Dropped {pid}: unknown source {source!r}.")
            continue
        if pieces:  # every piece after the base says where it goes
            where = item.get("place") if isinstance(item.get("place"), dict) else {}
            on = slugify(str(where.get("on") or ""), max_words=4, max_len=24)
            relation = str(where.get("relation") or "on_top").strip().lower()
            if on not in seen:
                notes.append(f"Dropped {pid}: it goes on {where.get('on')!r}, which is not an earlier piece.")
                continue
            if relation not in RELATIONS:
                notes.append(f"Dropped {pid}: unknown relation {relation!r}.")
                continue
            side = str(where.get("side") or "right").strip().lower()
            align = str(where.get("align") or "center").strip().lower()
            p["place"] = {"on": on, "relation": relation, "side": side if side in SIDES else "right",
                          "align": align if align in (*SIDES, "center") else "center"}
        pieces.append(p)
        seen.add(pid)
        if len(pieces) == MAX_PIECES:
            break
    return pieces, notes


def decide_from_report(report: dict, fallback_spec: str, rounds_asked: int, max_rounds: int | None = None) -> Decision:
    """The decision rules. Kept separate from the LLM call so they can be tested directly."""
    max_rounds = settings().gate_max_rounds if max_rounds is None else max_rounds
    scope = str(report.get("scope", "in_scope")).strip().lower()
    if scope not in SCOPES:
        scope = "in_scope"
    spec = str(report.get("spec") or "").strip() or fallback_spec
    unknowns = [u for u in report.get("unknowns") or [] if isinstance(u, dict)]
    critical = [u for u in unknowns if u.get("critical") is True]
    d = Decision(action=PROCEED, spec=spec, scope=scope, scope_reason=str(report.get("scope_reason") or ""),
                 assumptions=unknowns, name=slugify(str(report.get("name") or "")))
    d.search = " ".join(str(report.get("search") or "").split()) or d.name.replace("-", " ")
    if "examples" in report:
        picks = report["examples"] if isinstance(report["examples"], list) else []
        d.example_ids = list(dict.fromkeys(str(x).strip() for x in picks if str(x).strip()))

    if scope == "composite":
        d.pieces, notes = parse_pieces(report.get("pieces"))
        if len(d.pieces) < 2:  # not an assembly after all: one piece, or nothing usable
            only = d.pieces[0] if d.pieces else None
            if only and only["source"] == "cadquery":
                d.scope, d.spec = "in_scope", only["spec"]
            else:
                d.scope = "out_of_scope"
                d.search = only["search"][0] if only and only["search"] else d.search
            d.pieces = []
            notes.append("The piece plan had fewer than 2 usable pieces, so this is not built as a combination.")
        d.note = " ".join(notes)

    if d.scope == "out_of_scope":
        d.action = OUT_OF_SCOPE
    elif critical and rounds_asked < max_rounds:
        d.action = CLARIFY
        d.questions = [str(u.get("question") or f"What should {u.get('param')} be?") for u in critical][:MAX_QUESTIONS]
    elif critical:
        d.note = f"Still missing critical details after {rounds_asked} round(s); proceeding with best guesses."
    return d



class PassthroughGate:
    """gate = "off": the old behaviour, every request goes straight to generation."""

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        return Decision(action=PROCEED, spec="\n".join(t["content"] for t in conversation if t["role"] == "user"))


