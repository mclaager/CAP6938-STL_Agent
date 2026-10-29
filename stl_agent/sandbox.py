"""Sandboxed CadQuery executor + mesh validation for the text-to-STL agent.

Usage:
    from sandbox import run_cadquery, validate_mesh, feedback_for_llm

    res = run_cadquery(code, timeout=60)
    if res.success:
        report = validate_mesh(res.stl_path, expected_bbox=(60, 40, 5))
    print(feedback_for_llm(res, report))

Design notes:
- Generated code runs in a *separate Python process* with a hard timeout, so
  infinite loops, crashes in the OCC kernel (segfaults), or runaway memory
  can't take down the UI.
- The code may either call show_object(x) (the CQ-editor convention used in the
  RAG examples) or just leave the final shape in a variable named `result`.
- `language="sdf"` runs a script built with stl_agent.sdf (organic shapes, #42) instead of CadQuery.
- Tracebacks are trimmed to the frames from the generated code so the repair
  prompt contains signal, not 40 lines of CadQuery internals.
- This is a *safety-lite* sandbox (process isolation + timeout + memory cap on
  POSIX). It is not a security boundary against malicious code; fine for a
  course project where the code comes from your own LLM call.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Runner: executed inside the child process
# ---------------------------------------------------------------------------
from .config import FONT_DIR  # bundled Liberation Sans, used when text() has no fontPath

_RUNNER = textwrap.dedent(
    r'''
    import sys, json, traceback, linecache

    code_path, stl_path, result_path, mem_mb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    font_dir = sys.argv[5] if len(sys.argv) > 5 else None
    language = sys.argv[6] if len(sys.argv) > 6 else "cadquery"
    package_root = sys.argv[7] if len(sys.argv) > 7 else None

    # One OpenBLAS thread, set before numpy loads: OpenBLAS starts a thread per CPU, each reserving a large
    # buffer that counts against the cap below; on a 20-CPU machine meshing an SDF part then crashed (-11).
    # Our matrices are 3x3, so the threads never helped.
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Memory cap (POSIX only; silently skipped on Windows).
    # RLIMIT_DATA, not RLIMIT_AS: OpenCascade spawns one worker thread per CPU for booleans, and
    # their reserved (mostly unused) address space blows through an AS cap, hanging hole()/cut().
    try:
        import resource
        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    except Exception:
        pass

    source = open(code_path).read()
    FILENAME = "<generated>"
    linecache.cache[FILENAME] = (len(source), None, source.splitlines(True), FILENAME)

    shown = []
    def show_object(obj, *args, **kwargs):
        shown.append(obj)

    out = {"ok": False, "error_type": None, "error_msg": None, "traceback": None, "lib_call": None}

    def trimmed_tb():
        exc_type, exc, tb = sys.exc_info()
        frames = [f for f in traceback.extract_tb(tb) if f.filename == FILENAME]
        lines = []
        for f in frames:
            lines.append(f'  line {f.lineno}, in {f.name}: {(f.line or "").strip()}')
        return f"{exc_type.__name__}: {exc}\n" + "\n".join(lines)

    def lib_call():
        # The function the generated code called when the error happened (e.g. "Workplane.rect"):
        # the frame right after the last one in the generated code, if it is CadQuery's.
        tb, last_generated, after = sys.exc_info()[2], False, None
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_filename == FILENAME:
                last_generated, after = True, None
            elif last_generated and after is None:
                after = code
            tb = tb.tb_next
        if after is not None and "cadquery" in after.co_filename.replace("\\", "/"):
            return after.co_qualname
        return None

    def run_sdf():
        # SDF scripts (#42): `import sdf` / `from sdf import *` give stl_agent.sdf, and its names are also
        # predefined, as are math and numpy (np).
        sys.path.insert(0, package_root)
        from stl_agent import sdf
        import math
        import numpy as np

        sys.modules["sdf"] = sdf
        ns = {"__name__": "__generated__", "show_object": show_object, "math": math, "np": np, "sdf": sdf}
        ns.update({name: getattr(sdf, name) for name in sdf.__all__})
        exec(compile(source, FILENAME, "exec"), ns)
        obj = shown[-1] if shown else ns.get("result")
        if obj is None:
            raise ValueError("No shape produced: call show_object(shape) or assign the final shape to a variable named `result`.")
        if not isinstance(obj, sdf.SDF3):
            raise TypeError(f"show_object needs an sdf shape (sphere, capsule, ... joined with | - &), got {type(obj).__name__}.")
        sdf.save(obj, stl_path)

    try:
        if language == "sdf":
            run_sdf()
        else:
            import cadquery as cq

            # Text: CadQuery defaults to font="Arial", which is missing on most Linux systems, and
            # OpenCascade then renders every glyph as a box. Unless the code gives an explicit
            # fontPath, use the bundled Liberation Sans (metric-compatible with Arial).
            if font_dir:
                import inspect, os
                _make_text = cq.Compound.makeText
                _sig = inspect.signature(_make_text)
                _files = {"regular": "LiberationSans-Regular.ttf", "bold": "LiberationSans-Bold.ttf",
                          "italic": "LiberationSans-Italic.ttf"}

                def _make_text_with_font(*args, **kwargs):
                    bound = _sig.bind(*args, **kwargs)
                    if bound.arguments.get("fontPath") is None:
                        kind = bound.arguments.get("kind", "regular")
                        bound.arguments["fontPath"] = os.path.join(font_dir, _files.get(kind, _files["regular"]))
                    return _make_text(*bound.args, **bound.kwargs)

                cq.Compound.makeText = _make_text_with_font

            # math is allowed by the prompt; pre-load it so a forgotten import doesn't waste an attempt.
            import math
            ns = {"__name__": "__generated__", "show_object": show_object, "cq": cq, "math": math}
            exec(compile(source, FILENAME, "exec"), ns)

            obj = shown[-1] if shown else ns.get("result")
            if obj is None:
                raise ValueError("No shape produced: call show_object(shape) or assign the final shape to a variable named `result`.")

            # Normalise to something exportable
            if isinstance(obj, cq.Workplane):
                if not obj.vals():
                    raise ValueError("The final Workplane is empty (no solids). A previous operation likely removed all geometry.")
            cq.exporters.export(obj, stl_path)
        out["ok"] = True
    except BaseException as e:  # noqa: BLE001 - we want everything, incl. SystemExit
        out["error_type"] = type(e).__name__
        out["error_msg"] = str(e)
        out["traceback"] = trimmed_tb()
        try:
            out["lib_call"] = lib_call()
        except Exception:
            pass

    json.dump(out, open(result_path, "w"))
    '''
)


@dataclass
class ExecResult:
    success: bool
    stl_path: str | None = None
    error_type: str | None = None  # e.g. "AttributeError", "Timeout", "Crash"
    error_msg: str | None = None
    traceback: str | None = None  # trimmed to generated-code frames
    lib_call: str | None = None  # CadQuery function the error happened in, e.g. "Workplane.rect"
    stdout: str = ""
    stderr: str = ""
    code: str = ""


def run_cadquery(
    code: str,
    out_dir: str | Path | None = None,
    stl_name: str = "model.stl",
    timeout: float = 60.0,
    mem_mb: int = 2048,
    language: str = "cadquery",
) -> ExecResult:
    """Execute generated code in a subprocess and export an STL. `language`: "cadquery", or "sdf" for a
    script built with stl_agent.sdf (#42)."""
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="cadgen_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    code_path = out_dir / "generated.py"
    runner_path = out_dir / "_runner.py"
    result_path = out_dir / "_result.json"
    stl_path = out_dir / stl_name

    code_path.write_text(code)
    runner_path.write_text(_RUNNER)
    for p in (result_path, stl_path):
        p.unlink(missing_ok=True)

    cmd = [sys.executable, str(runner_path), str(code_path), str(stl_path), str(result_path), str(mem_mb), str(FONT_DIR), language,
           str(Path(__file__).resolve().parents[1])]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            success=False,
            error_type="Timeout",
            error_msg=f"Execution exceeded {timeout:.0f}s.",
            traceback=f"Timeout: code took longer than {timeout:.0f}s (likely an infinite loop or an extremely expensive boolean/fillet).",
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            code=code,
        )

    if not result_path.exists():
        # Child died before writing a result: segfault in OCC, OOM kill, etc.
        return ExecResult(
            success=False,
            error_type="Crash",
            error_msg=f"Process exited with code {proc.returncode} without a result.",
            traceback=f"Crash: the CAD kernel terminated abnormally (code {proc.returncode}). "
            "This often follows an invalid fillet/chamfer or boolean on degenerate geometry.\n"
            + (proc.stderr[-500:] if proc.stderr else ""),
            stdout=proc.stdout,
            stderr=proc.stderr,
            code=code,
        )

    info = json.loads(result_path.read_text())
    if info["ok"] and stl_path.exists() and stl_path.stat().st_size > 0:
        return ExecResult(True, str(stl_path), stdout=proc.stdout, stderr=proc.stderr, code=code)

    return ExecResult(
        success=False,
        error_type=info.get("error_type") or "ExportError",
        error_msg=info.get("error_msg") or "STL file was not produced.",
        traceback=info.get("traceback"),
        lib_call=info.get("lib_call"),
        stdout=proc.stdout,
        stderr=proc.stderr,
        code=code,
    )


# ---------------------------------------------------------------------------
# Mesh validation (your Week 1-2 geometric checks; also feeds the eval harness)
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    valid: bool
    watertight: bool = False
    winding_consistent: bool = False
    volume_mm3: float = 0.0
    bbox_mm: list[float] = field(default_factory=list)  # sorted-by-axis extents (x, y, z)
    n_components: int = 0
    n_faces: int = 0
    bbox_error_pct: list[float] | None = None  # per-axis % error vs expected, if given
    bbox_ok: bool | None = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def validate_mesh(
    stl_path: str,
    expected_bbox: tuple[float, float, float] | None = None,
    bbox_tol_pct: float = 5.0,
    match_axes_any_order: bool = True,
) -> ValidationReport:
    """Check printability and (optionally) dimensional accuracy.

    expected_bbox: requested overall (x, y, z) extents in mm. If
    match_axes_any_order is True, extents are compared after sorting, so a part
    built in a different orientation than you assumed isn't penalised.
    """
    import trimesh

    rep = ValidationReport(valid=False)
    try:
        mesh = trimesh.load(stl_path, force="mesh")
    except Exception as e:  # noqa: BLE001
        rep.issues.append(f"Could not load STL: {e}")
        return rep

    if mesh.is_empty or len(mesh.faces) == 0:
        rep.issues.append("Mesh is empty.")
        return rep

    # STL export leaves zero-area triangles at points like a sphere's poles or a cone's tip.
    # Each one would count as a separate piece and a hole, so drop them before checking.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    rep.n_faces = int(len(mesh.faces))
    rep.watertight = bool(mesh.is_watertight)
    rep.winding_consistent = bool(mesh.is_winding_consistent)
    rep.volume_mm3 = float(mesh.volume) if rep.watertight else 0.0
    rep.bbox_mm = [round(float(x), 4) for x in mesh.extents]
    rep.n_components = int(len(mesh.split(only_watertight=False)))

    if not rep.watertight:
        rep.issues.append("Mesh is not watertight (holes or non-manifold edges); not reliably printable.")
    if rep.watertight and rep.volume_mm3 <= 1e-6:
        rep.issues.append("Mesh has zero or negative volume.")
    if rep.n_components > 1:
        rep.issues.append(f"Mesh has {rep.n_components} disconnected pieces (a boolean may have left floating parts).")

    if expected_bbox is not None:
        got = sorted(rep.bbox_mm) if match_axes_any_order else rep.bbox_mm
        exp = sorted(expected_bbox) if match_axes_any_order else list(expected_bbox)
        errs = [abs(g - e) / e * 100 if e else 0.0 for g, e in zip(got, exp)]
        rep.bbox_error_pct = [round(x, 2) for x in errs]
        rep.bbox_ok = all(x <= bbox_tol_pct for x in errs)
        if not rep.bbox_ok:
            rep.issues.append(
                f"Bounding box {rep.bbox_mm} mm does not match expected {list(expected_bbox)} mm "
                f"(per-axis error % after axis sorting: {rep.bbox_error_pct})."
            )

    rep.valid = rep.watertight and rep.volume_mm3 > 1e-6 and rep.n_components == 1 and (rep.bbox_ok is not False)
    return rep


# ---------------------------------------------------------------------------
# Feedback text for the repair loop
# ---------------------------------------------------------------------------
def feedback_for_llm(res: ExecResult, report: ValidationReport | None = None, api_hints: bool = False) -> str:
    """Compact failure description to append to the next repair prompt. `api_hints` adds the real CadQuery
    signatures and names for the calls that failed (api_check.py)."""
    if not res.success:
        tb = res.traceback or res.error_msg or "unknown error"
        text = f"The code failed to execute.\n{tb}"
        if api_hints and res.code:
            from .api_check import hints

            extra = hints(res.code, res.error_type, res.error_msg, res.lib_call)
            if extra:
                text += "\n\n" + extra
        return text
    if report is not None and not report.valid:
        return "The code ran, but the resulting mesh failed validation:\n- " + "\n- ".join(report.issues)
    return ""
