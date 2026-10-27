"""Text-to-STL pipeline: RAG retrieval -> LLM code generation -> sandboxed execution
-> mesh validation -> self-repair loop.

Public entry points (used by the Gradio app):
    run_gate(conversation, rounds_asked) -> gate.Decision   (proceed / clarify / out_of_scope)
    run_pipeline(request, log, name=None) -> (stl_path, final_code, report_dict)
    run_fetch(candidate, log, name=None) -> (stl_path, report_dict)   (download from Thingiverse, #9)
    run_assembly(pieces, log, name=None, picks=None) -> (stl_path, code, report_dict)   (combined part, #10)

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

from .config import EXAMPLES_PATH, OUTPUT_DIR, ignored_env_vars, settings, settings_path
from .llm import make_llm, provider_config
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


# ---------------------------------------------------------------------------
# 4. App entry points
# ---------------------------------------------------------------------------
_retriever: BaseRetriever | None = None
_llm = None
_gate = None


def describe_settings() -> str:
    """One line for the app's startup log: settings file, provider and models, and ignored env variables."""
    s = settings()
    cfg = provider_config(s.provider)
    where = f"settings file {s.path}" if s.path else "built-in defaults (no settings file)"
    gate = f", gate {cfg['gate_model']}" if cfg["gate_model"] != cfg["model"] else ""
    line = f"Settings: {where}; provider {s.provider}, model {cfg['model']}{gate}, address {cfg['base_url'] or 'default'}."
    if ignored_env_vars():
        line += (f" Ignored environment variables (settings now live in {settings_path().name}): "
                 f"{', '.join(ignored_env_vars())}.")
    return line


def reset_backends() -> None:
    """Forget the cached LLM, gate and retriever, so the next request builds them from current settings."""
    global _retriever, _llm, _gate
    _retriever = _llm = _gate = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = make_llm()
    return _llm


def run_gate(conversation: list[dict], rounds_asked: int = 0):
    """Intake check before generation (see gate.py). Returns a gate.Decision.

    Thingiverse is optional: without THINGIVERSE_TOKEN, a plan that needs a downloaded piece is declined as
    out of scope, as before #9/#10 (no piece plan, no clarifying questions about a part that can't be built)."""
    global _gate
    if _gate is None:
        from .gate import make_gate

        cfg = provider_config(settings().provider)
        library = json.loads(EXAMPLES_PATH.read_text()) if settings().gate_picks else None
        # A separate gate model (e.g. Gemini's) has its own free-tier quota, so gate checks don't use up builds.
        llm = make_llm(model=cfg["gate_model"]) if cfg["gate_model"] != cfg["model"] else get_llm()
        _gate = make_gate(llm, library=library)
    return without_downloads_unless_token(_gate.decide(conversation, rounds_asked))


def without_downloads_unless_token(d):
    """Turn a plan with Thingiverse pieces into a plain out-of-scope decision when there is no token."""
    from .config import load_dotenv
    from .gate import OUT_OF_SCOPE
    from .sources import thingiverse

    load_dotenv()  # a token added to .env while the app runs counts
    if thingiverse.has_token() or not any(p["source"] == "thingiverse" for p in d.pieces):
        return d
    d.action, d.scope, d.pieces, d.questions = OUT_OF_SCOPE, "out_of_scope", [], []
    return d


def unique_stl_path(name: str, folder: Path | None = None) -> Path:
    """outputs/<name>.stl, or <name>-2.stl, -3, ... so earlier builds are kept."""
    folder = OUTPUT_DIR if folder is None else folder
    path, n = folder / f"{name}.stl", 2
    while path.exists():
        path, n = folder / f"{name}-{n}.stl", n + 1
    return path


def run_pipeline(request: str, log, name: str | None = None,
                 example_ids: list[str] | None = None) -> tuple[str, str, dict]:
    """`name` is a file-name slug (the gate supplies one); otherwise it's made from the request."""
    res = _generate(request, log, example_ids=example_ids)
    if res.unsupported or res.stl_path is None:
        raise RuntimeError(res.error or "Generation failed.")

    from .gate import slugify

    OUTPUT_DIR.mkdir(exist_ok=True)
    final_stl = unique_stl_path(slugify(name or "") or slugify(request) or "part")
    shutil.copy(res.stl_path, final_stl)
    log(f"Saved {final_stl.name}")
    log("Done." if res.success else "Done (with warnings; see report).")
    return str(final_stl), res.code or "", _panel_report(res)


def _generate(request: str, log, **kw) -> GenResult:
    """generate_cad with the app's cached LLM and retriever."""
    global _retriever
    if _retriever is None:
        _retriever = make_retriever()  # the retriever setting
    return generate_cad(request, get_llm(), _retriever, log=log, **kw)


def _panel_report(res: GenResult) -> dict:
    """The run's report without the long text kept for analysis (it stays in the run's report.json)."""
    report = res.report()
    report["attempt_log"] = [{k: v for k, v in h.items() if k not in ("reply", "reasoning", "feedback")}
                             for h in report["attempt_log"]]
    return report


def run_fetch(candidate, log, name: str | None = None) -> tuple[str, dict]:
    """Download a Thingiverse candidate (sources.thingiverse.Candidate) into outputs/ and validate it.

    Writes <name>.source.json next to the STL with the author, license and link, so the attribution
    stays with the file. Downloaded meshes are often not watertight: that is a warning, not a failure.
    """
    from .gate import slugify
    from .sources.thingiverse import download_stl

    OUTPUT_DIR.mkdir(exist_ok=True)
    stl = unique_stl_path(slugify(name or "") or slugify(candidate.name) or "thingiverse-part")
    log(f"Downloading '{candidate.name}' from Thingiverse...")
    _, others = download_stl(candidate.id, stl)
    log(f"Saved {stl.name}")
    source = {"site": "thingiverse", **candidate.to_dict(), "other_stl_files": others}
    stl.with_suffix(".source.json").write_text(json.dumps(source, indent=1))
    rep = validate_mesh(str(stl))
    log("Valid mesh." if rep.valid else "Mesh has issues (see report): " + "; ".join(rep.issues))
    report = {"success": rep.valid, "source": source, "validation": rep.to_dict()}
    return str(stl), report


# ---------------------------------------------------------------------------
# 5. Combined parts (#10)
# ---------------------------------------------------------------------------
def find_model(piece: dict, log, folder: Path):
    """Automatic pick for a Thingiverse piece: the first result, over its search phrases in order, that allows
    derivatives (the combined part is one) and downloads. Returns (candidate, stl path) or (None, None)."""
    from .sources import thingiverse as tv

    for phrase in piece.get("search") or []:
        log(f"{piece['id']}: searching Thingiverse for '{phrase}'...")
        for c in tv.search(phrase):
            if not c.allows_derivatives:
                continue
            try:
                path, _ = tv.download_stl(c.id, folder / f"{piece['id']}.stl")
            except tv.ThingiverseError as e:
                log(f"{piece['id']}: skipped '{c.name}' ({e})")
                continue
            return c, path
    return None, None


def run_assembly(pieces: list[dict], log, name: str | None = None,
                 picks: dict | None = None) -> tuple[str, str, dict]:
    """Make each piece of a gate plan (gate.base.parse_pieces), place them and merge them into one STL.

    `picks`: piece id -> the user's Thingiverse candidate, or None when a search found nothing (the piece is
    then built from its fallback_spec). Pieces not in `picks` are picked automatically (find_model).
    Thingiverse pieces are fetched first, so a missing model costs no generation requests.
    """
    import trimesh

    from .gate import slugify
    from .sources import assembly
    from .sources import thingiverse as tv

    picks = picks or {}
    plan = [dict(p) for p in pieces]
    folder = OUTPUT_DIR / time.strftime("assembly_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    rows = {p["id"]: {"id": p["id"], "source": p["source"]} for p in plan}
    paths: dict[str, Path] = {}
    credits: list[dict] = []
    codes: list[str] = []

    for p in plan:  # 1. downloaded pieces
        if p["source"] != "thingiverse":
            continue
        if p["id"] in picks:
            c, path = picks[p["id"]], None
            if c is not None:
                log(f"{p['id']}: downloading '{c.name}'...")
                path, _ = tv.download_stl(c.id, folder / f"{p['id']}.stl")
        else:
            c, path = find_model(p, log, folder)
        if c is None:
            if not p.get("fallback_spec"):
                raise RuntimeError(f"No Thingiverse model found for the {p['id']}, and the plan has no simplified "
                                   "version to build instead.")
            log(f"{p['id']}: no Thingiverse model found; building a simplified one instead.")
            p.update(source="cadquery", spec=p["fallback_spec"], fallback=True)
            rows[p["id"]]["source"] = "cadquery (no Thingiverse model found)"
            continue
        paths[p["id"]] = path
        credits.append({"piece": p["id"], **c.to_dict()})
        rows[p["id"]]["model"] = c.credit()
        codes.append(f"# --- {p['id']}: downloaded, not generated ---\n# {c.credit()}")

    for p in plan:  # 2. generated pieces
        if p["source"] != "cadquery":
            continue
        log(f"{p['id']}: generating with CadQuery...")
        res = _generate(p["spec"], log, out_dir=folder / p["id"])
        if res.unsupported or res.stl_path is None:
            raise RuntimeError(f"Could not build the {p['id']}: {res.error or 'generation failed.'}")
        paths[p["id"]] = Path(res.stl_path)
        rows[p["id"]]["attempts"] = res.attempts
        if not res.success:  # a best-effort piece: say so, it may be what looks wrong in the result
            rows[p["id"]]["warning"] = res.error
            log(f"{p['id']}: {res.error}")
        codes.append(f"# --- {p['id']} ---\n{res.code or ''}")

    log("Placing and merging the pieces...")
    meshes = {}
    for p in plan:
        m = trimesh.load(paths[p["id"]], force="mesh")
        # Downloaded models (and their stand-ins) come in any size: scale them to the planned height.
        if p.get("height_mm") and (p["source"] == "thingiverse" or p.get("fallback")):
            m = assembly.scale_to_height(m, p["height_mm"])
        meshes[p["id"]] = m
    placed = assembly.place(meshes, plan)
    ids = [p["id"] for p in plan]
    merged, notes = assembly.merge([placed[i] for i in ids], ids)
    for i in ids:
        rows[i]["size_mm"] = [round(float(x), 1) for x in placed[i].extents]
        rows[i]["watertight"] = bool(placed[i].is_watertight)
    for n in notes:
        log(n)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stl = unique_stl_path(slugify(name or "") or "-".join(ids)[:48] or "combined-part")
    merged.export(stl)
    if credits:
        stl.with_suffix(".source.json").write_text(json.dumps({"site": "thingiverse", "pieces": credits}, indent=1))
    val = validate_mesh(str(stl))
    log(f"Saved {stl.name}")
    log("Done." if val.valid else "Done (with warnings; see report).")
    report = {"success": val.valid, "pieces": list(rows.values()), "merge_notes": notes,
              "validation": val.to_dict(), "folder": str(folder)}
    return str(stl), "\n\n".join(codes) + "\n", report
