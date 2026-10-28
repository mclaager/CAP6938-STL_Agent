"""Generation core: prompts, reply parsing, and the generate -> execute -> validate -> repair loop.

    generate_cad(request, llm, retriever, ...) -> GenResult

Used by the app (through pipeline.py, which adds cached backends and saving) and by the benchmark
(eval/bench.py). Pass any object with `.complete(system, messages) -> str` as the LLM (see llm.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import OUTPUT_DIR, settings
from .retrieval import BaseRetriever
from .sandbox import ExecResult, ValidationReport, feedback_for_llm, run_cadquery, validate_mesh

# ---------------------------------------------------------------------------
# 1. Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a CAD engineer that turns a plain-English part description into a CadQuery (Python) script.
The script is executed, exported to STL, and checked automatically. Units are millimetres.

OUTPUT FORMAT (follow exactly)
PLAN:
- 2-6 short bullets: base shape, features in order, key dimensions, any assumption you made.
BBOX: X, Y, Z
```python
<complete script>
```
BBOX is the overall bounding-box size of the FINISHED part in mm (three numbers, any axis order).

If the request cannot reasonably be built from CAD primitives and operations (organic or artistic
shapes, characters, logos, anything needing a mesh sculpt) reply with ONE line instead:
UNSUPPORTED: <short reason>
If a request is merely underspecified, pick sensible dimensions, state them in PLAN, and proceed.

SCRIPT RULES
- Only `import cadquery as cq` (and `math` if needed). No file I/O, no other imports.
- Put every dimension in a named variable at the top (mm); derive related values from them.
- The final solid must be passed to `show_object(result)` and be a single connected solid.
- Prefer the fluent API: Workplane(...).box / circle / rect / polyline -> extrude / cutBlind / hole.
- Select faces/edges with string selectors: faces(">Z"), edges("|Z"), edges("#Z"), faces("<X").
  After faces(">Z") call .workplane() before placing holes/sketches on it.
- Holes: .hole(d) / .cboreHole(d, cbore_d, cbore_depth) / .cskHole(d, csk_d, csk_angle);
  place several with .pushPoints([...]), .rarray(xs, ys, nx, ny), or .polarArray(radius, 0, 360, n).
- Hollowing: .faces(">Z").shell(-wall) (negative = inward, selected face is opened).
- Fillet/chamfer radius must be clearly smaller than the thinnest adjacent feature, and apply it
  to the whole solid BEFORE cutting small holes next to those edges when possible.
- Centering: box() and rect() are centred on the workplane origin unless centered=(False, ...).
  extrude() has NO `centered` argument; use extrude(h, both=True) to extrude symmetrically.
- Do NOT invent methods (there is no fillet_all, no add_hole, no Workplane.sphere with 3 args).
  Methods that exist include: box, sphere, cylinder, circle, rect, polygon, ellipse, slot2D,
  extrude, revolve, loft, sweep, union, cut, intersect, translate, rotate, mirror, fillet, chamfer.

Learn the style from the examples, but adapt geometry and dimensions to the new request rather
than copying. When fixing an error, change only what is needed and reply in the same format.
"""


def format_examples(examples: list[dict]) -> str:
    blocks = []
    for i, e in enumerate(examples, 1):
        bb = ", ".join(f"{x:g}" for x in e["expected_bbox_mm"])
        blocks.append(
            f"### Example {i}\nRequest: {e['request']}\nBBOX: {bb}\n```python\n{e['code'].strip()}\n```"
        )
    return "\n\n".join(blocks)


def build_first_user_message(request: str, examples: list[dict]) -> str:
    if not examples:  # RAG off (e.g. the benchmark's no-RAG baseline)
        return f"Handle this request. Reply in the required format.\n\nRequest: {request}"
    return (
        "Here are solved examples of similar requests:\n\n"
        f"{format_examples(examples)}\n\n"
        "---\nNow handle this request. Reply in the required format.\n\n"
        f"Request: {request}"
    )


def build_repair_message(feedback: str) -> str:
    return (
        f"{feedback}\n\nFix the script. Keep the original intent and dimensions. "
        "Reply with PLAN, BBOX and the complete corrected script in the same format."
    )


# ---------------------------------------------------------------------------
# 2. Response parsing
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    code: str | None = None
    bbox: tuple[float, float, float] | None = None
    plan: str = ""
    unsupported: str | None = None


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_BBOX_RE = re.compile(r"BBOX\s*:\s*\(?\s*([\d.eE+-]+)\s*[,xX×]\s*([\d.eE+-]+)\s*[,xX×]\s*([\d.eE+-]+)", re.I)
_UNSUP_RE = re.compile(r"^\s*UNSUPPORTED\s*:\s*(.+)$", re.M)


def parse_response(text: str) -> Parsed:
    p = Parsed()
    m = _UNSUP_RE.search(text)
    if m and not _CODE_RE.search(text):
        p.unsupported = m.group(1).strip()
        return p
    # take the LAST code block (models sometimes show a broken one, then the fix)
    blocks = _CODE_RE.findall(text)
    if blocks:
        p.code = blocks[-1].strip() + "\n"
    m = _BBOX_RE.search(text)
    if m:
        try:
            p.bbox = tuple(float(g) for g in m.groups())  # type: ignore[assignment]
        except ValueError:
            pass
    pm = re.search(r"PLAN\s*:(.*?)(?:BBOX\s*:|```|$)", text, re.S | re.I)
    if pm:
        p.plan = pm.group(1).strip()
    return p


# ---------------------------------------------------------------------------
# 3. Core loop
# ---------------------------------------------------------------------------
@dataclass
class GenResult:
    success: bool
    stl_path: str | None = None
    code: str | None = None
    attempts: int = 0
    unsupported: str | None = None
    validation: ValidationReport | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # per-attempt record, for error analysis
    error: str | None = None
    seconds: float = 0.0

    def report(self) -> dict:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "unsupported": self.unsupported,
            "retrieved_examples": self.retrieved_ids,
            "validation": self.validation.to_dict() if self.validation else None,
            "attempt_log": self.history,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "stl_path": self.stl_path,  # final (or best-effort) part
        }


def generate_cad(
    request: str,
    llm,
    retriever: BaseRetriever,
    log=lambda s: None,
    out_dir: str | Path | None = None,
    max_attempts: int | None = None,
    top_k: int | None = None,
    use_rag: bool = True,
    api_hints: bool | None = None,
    example_ids: list[str] | None = None,
    exec_timeout: float = 60.0,
) -> GenResult:
    """Run the full generate -> execute -> validate -> repair loop. `max_attempts` / `top_k` / `api_hints` left
    as None come from the current settings."""
    max_attempts = max_attempts or settings().max_attempts
    api_hints = settings().api_hints if api_hints is None else api_hints
    top_k = settings().top_k if top_k is None else top_k
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = GenResult(success=False)

    # Examples: the intake gate's picks when it made any (gate_picks setting), otherwise the retriever
    # run on the request (in the app, the gate's confirmed spec).
    by_id = {e["id"]: e for e in retriever.examples}
    picked = [by_id[i] for i in (example_ids or []) if i in by_id][:top_k]
    if not use_rag:
        examples = []
    elif picked:
        examples = picked
    else:
        examples = retriever.retrieve(request, top_k)
    res.retrieved_ids = [e["id"] for e in examples]
    source = "chosen by the intake gate" if use_rag and picked else "retrieved"
    log(f"Using {len(examples)} example(s) {source}: {', '.join(res.retrieved_ids) or 'none'}")

    messages = [{"role": "user", "content": build_first_user_message(request, examples)}]
    best: tuple[ExecResult, ValidationReport, str] | None = None  # best-effort fallback

    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        log(f"Attempt {attempt}/{max_attempts}: asking the model for CadQuery code...")
        reply = llm.complete(SYSTEM_PROMPT, messages)
        messages.append({"role": "assistant", "content": reply})
        parsed = parse_response(reply)
        # The full reply and reasoning are kept for error analysis (the code is also stored on its own below).
        rec = {"attempt": attempt, "plan": parsed.plan, "bbox_claimed": parsed.bbox, "reply": reply,
               "reasoning": getattr(llm, "last_reasoning", "") or ""}

        if parsed.unsupported:
            res.unsupported = parsed.unsupported
            res.error = f"Request not supported by parametric CAD generation: {parsed.unsupported}"
            rec["outcome"] = "unsupported"
            res.history.append(rec)
            log(f"Model declined: {parsed.unsupported}")
            break

        if not parsed.code:
            rec["outcome"] = "no_code_block"
            res.history.append(rec)
            if llm.last_truncated:
                rec["outcome"] = "truncated"
                log("Reply was cut off by the output token limit before the code finished; asking again.")
            else:
                log("Reply contained no code block; asking again.")
            rec["feedback"] = "Your reply had no ```python code block. Reply in the required format."
            messages.append({"role": "user", "content": rec["feedback"]})
            continue

        log("Executing code in sandbox...")
        ex = run_cadquery(parsed.code, out_dir=out_dir / f"attempt_{attempt}", timeout=exec_timeout)
        rec["code"] = parsed.code

        if not ex.success:
            rec.update(outcome="exec_error", error_type=ex.error_type, error=(ex.traceback or ex.error_msg))
            if api_hints:
                from .api_check import check_code

                rec["api_problems"] = len(check_code(parsed.code))
            res.history.append(rec)
            log(f"Execution failed ({ex.error_type}): {(ex.error_msg or '')[:160]}")
            rec["feedback"] = build_repair_message(feedback_for_llm(ex, api_hints=api_hints))
            messages.append({"role": "user", "content": rec["feedback"]})
            continue

        log("Validating mesh (watertight, volume, bounding box)...")
        val = validate_mesh(ex.stl_path, expected_bbox=parsed.bbox)
        rec.update(outcome="ok" if val.valid else "invalid_mesh", validation=val.to_dict())
        res.history.append(rec)
        if best is None or (val.valid and not best[1].valid) or (val.watertight and not best[1].watertight):
            best = (ex, val, parsed.code)

        if val.valid:
            log(f"Valid mesh: bbox {val.bbox_mm} mm, volume {val.volume_mm3:.0f} mm^3.")
            res.success, res.stl_path, res.code, res.validation = True, ex.stl_path, parsed.code, val
            break

        log("Mesh failed validation: " + "; ".join(val.issues))
        rec["feedback"] = build_repair_message(feedback_for_llm(ex, val))
        messages.append({"role": "user", "content": rec["feedback"]})

    if not res.success and best is not None and not res.unsupported:
        ex, val, code = best
        res.stl_path, res.code, res.validation = ex.stl_path, code, val
        res.error = "Best-effort result: mesh did not pass all checks after all attempts."
        log(res.error)
    elif not res.success and not res.error:
        last = res.history[-1] if res.history else {}
        res.error = f"No valid result after {res.attempts} attempt(s). Last failure: " + str(
            last.get("error") or last.get("outcome") or "no code produced"
        )

    res.seconds = time.time() - t0
    (out_dir / "report.json").write_text(json.dumps(res.report(), indent=1, default=str))
    return res
