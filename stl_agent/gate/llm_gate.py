"""Gate backed by a general LLM: it reports scope, a filled-in spec and unknowns as JSON."""

from __future__ import annotations

import json

from ..replies import parse_json_object
from .base import PROCEED, Decision, decide_from_report

GATE_PROMPT = """\
You are the intake step of a text-to-CAD agent. A later step writes CadQuery (Python parametric CAD)
code for the part, so it can build anything made from extrude, revolve, loft, sweep, boolean
union/cut, holes, fillets, chamfers, shells, patterns and simple text. Units are millimetres.

Read the conversation and reply with ONE JSON object in a ```json block, nothing else:
{
  "scope": "in_scope" | "simplified" | "out_of_scope" | "composite",
  "scope_reason": "one short sentence; empty when in_scope",
  "spec": "complete, self-contained description of the part to build, with every dimension in mm, including everything the user said in later messages",
  "name": "short file name for the part: 2-5 lowercase words joined by hyphens, e.g. flange-80mm-6-holes",
  "search": "2-4 plain words to find an existing 3D model of this object on a model site, no sizes, e.g. horse statue",
  "organic_spec": "only for a single organic or sculpted object (out_of_scope): what it looks like, for sculpting it from rounded shapes: pose, main body parts and their proportions, overall height in mm; empty otherwise, and for logos, pictures and non-objects",
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
- composite: one part made by joining several distinct objects, where at least one of them is
  organic or sculpted (a cat sitting on a tree, a horse statue on a plinth with a name plate, people
  on a bench). Sculpted pieces are downloaded as existing models; geometric pieces are built with
  code. Check composite before out_of_scope: a request that combines objects is composite even when
  some of them are people or animals. If every piece could be built with code, use in_scope instead:
  one script is better than joined pieces.
- out_of_scope: a single organic or sculpted object with nothing to join it to (a person, an animal,
  a face, a character, a realistic plant), copies of images or logos, or something that is not a
  physical part at all.

COMPOSITE (only when scope is composite): add a "pieces" list, in build order, the base first:
  "pieces": [
    {"id": "tree", "source": "thingiverse", "search": ["oak tree", "tree"], "height_mm": 120,
     "fallback_spec": "stylized tree: 15 mm cylinder trunk 50 mm tall under a 70 mm sphere canopy",
     "organic_spec": "oak tree 120 mm tall: thick trunk splitting into two branches, a broad rounded canopy"},
    {"id": "cat", "source": "thingiverse", "search": ["sitting cat", "cat"], "height_mm": 30,
     "fallback_spec": "stylized sitting cat: ...", "organic_spec": "sitting cat 30 mm tall, tail curled around its front paws, ears up",
     "place": {"on": "tree", "relation": "on_top"}},
    {"id": "plate", "source": "cadquery", "spec": "40x12x3 mm plate with the raised text TOM",
     "place": {"on": "tree", "relation": "beside", "side": "front"}}
  ]
- source: "thingiverse" for organic/sculpted pieces (give "search": 1-3 short phrases, most specific
  first, "fallback_spec": a simplified geometric version to build if no model is found, and
  "organic_spec": what it looks like, as for a single organic object), or
  "cadquery" for geometric pieces (give "spec" with every dimension in mm).
- height_mm: the piece's height; downloaded models are scaled to it.
- place (every piece after the first): "on" is an earlier piece's id; "relation" is "on_top" (rests
  on it), "under" (the other piece rests on this one), "beside" (with "side": left, right, front or
  back) or "inside"; "align" (on_top only): center, left, right, front or back.
- spec: describe the whole combination in words.

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


LIBRARY_PROMPT = """
EXAMPLE LIBRARY
The code-writing step can be shown up to 3 solved examples from this library. Add two fields to
the JSON object:
  "examples": ["id", ...]
For "examples", pick up to 3 ids whose geometry or CadQuery operations (revolve, loft, shell,
hole patterns, slots, text, sweep, angled plates, ...) are closest to the part in your spec, most
useful first. Judge by how the part is built, not by its purpose. Use [] if none is close.

{index}
"""


def library_index(examples: list[dict]) -> str:
    return "\n".join(f"{e['id']}: {e['request']}" for e in examples)


def _transcript(conversation: list[dict]) -> str:
    lines = []
    for turn in conversation:
        who = "User" if turn["role"] == "user" else "Agent"
        lines.append(f"{who}: {turn['content'].strip()}")
    return "Conversation so far:\n\n" + "\n\n".join(lines)


def parse_gate_reply(text: str) -> dict:
    """Pull the JSON object out of the reply (fenced block preferred, else the outermost braces)."""
    return parse_json_object(text)



class LLMGate:
    def __init__(self, llm, max_rounds: int | None = None, library: list[dict] | None = None):
        """With `library`, the gate also picks relevant examples. `max_rounds` None: the gate_max_rounds setting."""
        self.llm, self.max_rounds = llm, max_rounds
        self.known_ids = {e["id"] for e in library} if library else set()
        self.prompt = GATE_PROMPT + (LIBRARY_PROMPT.format(index=library_index(library)) if library else "")

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        fallback_spec = "\n".join(t["content"] for t in conversation if t["role"] == "user")
        reply = self.llm.complete(self.prompt, [{"role": "user", "content": _transcript(conversation)}])
        try:
            report = parse_gate_reply(reply)
        except (ValueError, json.JSONDecodeError):
            # Fail open: a broken gate reply should not block generation.
            return Decision(action=PROCEED, spec=fallback_spec, note="Gate reply was not valid JSON; using the raw request.")
        d = decide_from_report(report, fallback_spec, rounds_asked, self.max_rounds)
        if d.example_ids is not None:
            # Drop ids the model made up; keep the order it gave.
            d.example_ids = [i for i in d.example_ids if i in self.known_ids][:3] if self.known_ids else None
        return d
