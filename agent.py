"""Text-to-STL agent: RAG retrieval -> LLM code generation -> sandboxed execution
-> mesh validation -> self-repair loop.

Public entry point (matches the Gradio app's contract):
    run_pipeline(request, log) -> (stl_path, final_code, report_dict)

Config via environment variables:
    ANTHROPIC_API_KEY   required for the real LLM
    CAD_MODEL           model id (default: claude-sonnet-5-5)
    CAD_MAX_ATTEMPTS    generation + repair attempts (default 4)
    CAD_TOP_K           RAG examples per prompt (default 3)

Swap the LLM by passing any object with `.complete(system, messages) -> str`
(see AnthropicLLM / ScriptedLLM below); the eval harness can reuse `generate_cad`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from sandbox import ExecResult, ValidationReport, feedback_for_llm, run_cadquery, validate_mesh

HERE = Path(__file__).parent
EXAMPLES_PATH = HERE / "examples.json"
OUTPUT_DIR = HERE / "outputs"

MODEL = os.environ.get("CAD_MODEL", "claude-sonnet-5-5")
MAX_ATTEMPTS = int(os.environ.get("CAD_MAX_ATTEMPTS", "4"))
TOP_K = int(os.environ.get("CAD_TOP_K", "3"))


# ---------------------------------------------------------------------------
# 1. Retrieval (RAG over the example library)
# ---------------------------------------------------------------------------
class ExampleRetriever:
    """TF-IDF similarity over example *request text only* (not code).

    TF-IDF needs no model download and works offline, which is plenty for a
    ~30-example library. To try dense embeddings later, replace `_vectorize`
    and `retrieve` with sentence-transformers; the interface stays the same.
    A per-class cap keeps the top-k diverse instead of three near-duplicates.
    """

    def __init__(self, examples: list[dict], max_per_class: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.examples = examples
        self.max_per_class = max_per_class
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([e["request"] for e in examples])

    @classmethod
    def from_file(cls, path: Path = EXAMPLES_PATH, **kw) -> "ExampleRetriever":
        return cls(json.loads(Path(path).read_text()), **kw)

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        picked, per_class = [], {}
        for i in order:
            ex = self.examples[i]
            if per_class.get(ex["cls"], 0) >= self.max_per_class:
                continue
            picked.append(ex)
            per_class[ex["cls"]] = per_class.get(ex["cls"], 0) + 1
            if len(picked) == k:
                break
        return picked


# ---------------------------------------------------------------------------
# 2. Prompts
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
# 3. Response parsing
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
# 4. LLM backends
# ---------------------------------------------------------------------------
class AnthropicLLM:
    def __init__(self, model: str = MODEL, max_tokens: int = 2048, temperature: float | None = None):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.input_tokens = self.output_tokens = 0

    def complete(self, system: str, messages: list[dict]) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        # Newer SDK/model versions don't accept sampling params; only send if explicitly requested.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class ScriptedLLM:
    """Returns canned replies in order. For tests and offline demos."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), []

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


# ---------------------------------------------------------------------------
# 5. Core loop
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
    retriever: ExampleRetriever,
    log=lambda s: None,
    out_dir: str | Path | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    top_k: int = TOP_K,
    use_rag: bool = True,
    exec_timeout: float = 60.0,
) -> GenResult:
    """Run the full generate -> execute -> validate -> repair loop."""
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = GenResult(success=False)

    examples = retriever.retrieve(request, top_k) if use_rag else []
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
# 6. Gradio contract
# ---------------------------------------------------------------------------
_retriever: ExampleRetriever | None = None
_llm = None


def run_pipeline(request: str, log) -> tuple[str, str, dict]:
    global _retriever, _llm
    if _retriever is None:
        _retriever = ExampleRetriever.from_file()
    if _llm is None:
        _llm = AnthropicLLM()

    res = generate_cad(request, _llm, _retriever, log=log)
    if res.unsupported or res.stl_path is None:
        raise RuntimeError(res.error or "Generation failed.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    final_stl = OUTPUT_DIR / "latest.stl"
    shutil.copy(res.stl_path, final_stl)
    log("Done." if res.success else "Done (with warnings; see report).")
    return str(final_stl), res.code or "", res.report()
