"""App entry points: the intake gate, building a part, Thingiverse downloads and combined parts.

    run_gate(conversation, rounds_asked) -> gate.Decision   (proceed / clarify / out_of_scope)
    run_pipeline(request, log, name=None) -> (stl_path, final_code, report_dict)
    run_fetch(candidate, log, name=None) -> (stl_path, report_dict)   (download from Thingiverse, #9)
    run_assembly(pieces, log, name=None, picks=None) -> (stl_path, code, report_dict)   (combined part, #10)

The LLM, gate and retriever are built from the current settings on first use and cached until
reset_backends(). The generation loop itself is in generate.py; finished parts are saved to outputs/.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .config import EXAMPLES_PATH, OUTPUT_DIR, ignored_env_vars, settings, settings_path
from .generate import GenResult, generate_cad
from .llm import make_llm, provider_config
from .retrieval import BaseRetriever, make_retriever
from .sandbox import validate_mesh

# ---------------------------------------------------------------------------
# Cached backends and app entry points
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
# Combined parts (#10)
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
    folder = OUTPUT_DIR / "runs" / time.strftime("assembly_%Y%m%d_%H%M%S")
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
