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
_RUNNER = textwrap.dedent(
    r'''
    import sys, json, traceback, linecache

    code_path, stl_path, result_path, mem_mb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    # Memory cap (POSIX only; silently skipped on Windows)
    try:
        import resource
        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass

    source = open(code_path).read()
    FILENAME = "<generated>"
    linecache.cache[FILENAME] = (len(source), None, source.splitlines(True), FILENAME)

    shown = []
    def show_object(obj, *args, **kwargs):
        shown.append(obj)

    out = {"ok": False, "error_type": None, "error_msg": None, "traceback": None}

    def trimmed_tb():
        exc_type, exc, tb = sys.exc_info()
        frames = [f for f in traceback.extract_tb(tb) if f.filename == FILENAME]
        lines = []
        for f in frames:
            lines.append(f'  line {f.lineno}, in {f.name}: {(f.line or "").strip()}')
        return f"{exc_type.__name__}: {exc}\n" + "\n".join(lines)

    try:
        import cadquery as cq
        ns = {"__name__": "__generated__", "show_object": show_object, "cq": cq}
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
    stdout: str = ""
    stderr: str = ""
    code: str = ""


def run_cadquery(
    code: str,
    out_dir: str | Path | None = None,
    stl_name: str = "model.stl",
    timeout: float = 60.0,
    mem_mb: int = 2048,
) -> ExecResult:
    """Execute generated CadQuery code in a subprocess and export an STL."""
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

    cmd = [sys.executable, str(runner_path), str(code_path), str(stl_path), str(result_path), str(mem_mb)]
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
def feedback_for_llm(res: ExecResult, report: ValidationReport | None = None) -> str:
    """Compact failure description to append to the next repair prompt."""
    if not res.success:
        tb = res.traceback or res.error_msg or "unknown error"
        return f"The code failed to execute.\n{tb}"
    if report is not None and not report.valid:
        return "The code ran, but the resulting mesh failed validation:\n- " + "\n- ".join(report.issues)
    return ""


if __name__ == "__main__":
    # Smoke test: one good snippet, one with a bug, one that hangs.
    good = "import cadquery as cq\nresult = cq.Workplane('XY').box(60, 40, 5).faces('>Z').workplane().hole(4)\nshow_object(result)\n"
    bad = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10).fillet_all(1)\n"
    hang = "while True:\n    pass\n"

    for name, snippet, kw in [("good", good, {"expected_bbox": (60, 40, 5)}), ("bad", bad, {}), ("hang", hang, {})]:
        r = run_cadquery(snippet, timeout=10)
        print(f"--- {name}: success={r.success}")
        if r.success:
            print(validate_mesh(r.stl_path, **kw).to_dict())
        else:
            print(feedback_for_llm(r))
