"""Text-to-STL pipeline: RAG retrieval -> LLM code generation -> sandboxed execution
-> mesh validation -> self-repair loop.

Public entry points (used by the Gradio app):
    run_gate(conversation, rounds_asked) -> gate.Decision   (proceed / clarify / out_of_scope)
    run_pipeline(request, log, name=None) -> (stl_path, final_code, report_dict)

Settings come from environment variables / `.env` (see config.py). Swap the LLM by passing any
object with `.complete(system, messages) -> str` (see llm.py); the eval harness can reuse `generate_cad`.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import GATE_MODEL, MAX_ATTEMPTS, OUTPUT_DIR, PROVIDER, RAG_MIN_SIM, TOP_K
from .llm import PROVIDERS, make_llm
from .retrieval import BaseRetriever, make_retriever
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
- Centering: box() and circle() are centred on the workplane origin unless centered=(False, ...).
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
        }


def generate_cad(
    request: str,
    llm,
    retriever: BaseRetriever,
    log=lambda s: None,
    out_dir: str | Path | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    top_k: int = TOP_K,
    use_rag: bool = True,
    min_sim: float = RAG_MIN_SIM,
    exec_timeout: float = 60.0,
) -> GenResult:
    """Run the full generate -> execute -> validate -> repair loop."""
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = GenResult(success=False)

    examples = retriever.retrieve(request, top_k, min_sim) if use_rag else []
    res.retrieved_ids = [e["id"] for e in examples]
    log(f"Retrieved {len(examples)} similar example(s): {', '.join(res.retrieved_ids) or 'none'}")

    messages = [{"role": "user", "content": build_first_user_message(request, examples)}]
    best: tuple[ExecResult, ValidationReport, str] | None = None  # best-effort fallback

    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        log(f"Attempt {attempt}/{max_attempts}: asking the model for CadQuery code...")
        reply = llm.complete(SYSTEM_PROMPT, messages)
        messages.append({"role": "assistant", "content": reply})
        parsed = parse_response(reply)
        rec = {"attempt": attempt, "plan": parsed.plan, "bbox_claimed": parsed.bbox}

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
            if getattr(llm, "last_truncated", False):
                rec["outcome"] = "truncated"
                log("Reply was cut off by the output token limit before the code finished; asking again.")
            else:
                log("Reply contained no code block; asking again.")
            messages.append({"role": "user", "content": "Your reply had no ```python code block. Reply in the required format."})
            continue

        log("Executing code in sandbox...")
        ex = run_cadquery(parsed.code, out_dir=out_dir / f"attempt_{attempt}", timeout=exec_timeout)
        rec["code"] = parsed.code

        if not ex.success:
            rec.update(outcome="exec_error", error_type=ex.error_type, error=(ex.traceback or ex.error_msg))
            res.history.append(rec)
            log(f"Execution failed ({ex.error_type}): {(ex.error_msg or '')[:160]}")
            messages.append({"role": "user", "content": build_repair_message(feedback_for_llm(ex))})
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
        messages.append({"role": "user", "content": build_repair_message(feedback_for_llm(ex, val))})

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


# ---------------------------------------------------------------------------
# 4. App entry points
# ---------------------------------------------------------------------------
_retriever: BaseRetriever | None = None
_llm = None
_gate = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = make_llm()
    return _llm


def run_gate(conversation: list[dict], rounds_asked: int = 0):
    """Intake check before generation (see gate.py). Returns a gate.Decision."""
    global _gate
    if _gate is None:
        from .gate import make_gate

        gate_model = GATE_MODEL or PROVIDERS.get(PROVIDER, {}).get("default_gate_model")
        _gate = make_gate(make_llm(model=gate_model) if gate_model else get_llm())
    return _gate.decide(conversation, rounds_asked)


def unique_stl_path(name: str, folder: Path = OUTPUT_DIR) -> Path:
    """outputs/<name>.stl, or <name>-2.stl, -3, ... so earlier builds are kept."""
    path, n = folder / f"{name}.stl", 2
    while path.exists():
        path, n = folder / f"{name}-{n}.stl", n + 1
    return path


def run_pipeline(request: str, log, name: str | None = None) -> tuple[str, str, dict]:
    """`name` is a file-name slug (the gate supplies one); otherwise it's made from the request."""
    global _retriever
    if _retriever is None:
        _retriever = make_retriever()  # CAD_RETRIEVER

    res = generate_cad(request, get_llm(), _retriever, log=log)
    if res.unsupported or res.stl_path is None:
        raise RuntimeError(res.error or "Generation failed.")

    from .gate import slugify

    OUTPUT_DIR.mkdir(exist_ok=True)
    final_stl = unique_stl_path(slugify(name or "") or slugify(request) or "part")
    shutil.copy(res.stl_path, final_stl)
    log(f"Saved {final_stl.name}")
    log("Done." if res.success else "Done (with warnings; see report).")
    return str(final_stl), res.code or "", res.report()
