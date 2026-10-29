"""SDF modeling for organic shapes (#42): the library, meshing, the sandbox, generation, the example library and
routing organic objects and pieces to it. No LLM requests."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from stl_agent import sdf
from stl_agent.sdf import X, Y, Z, box, capsule, ellipsoid, sphere, union


def at(shape, *points):
    return shape(np.array(points, dtype=float))


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
def test_distances_are_negative_inside_and_zero_on_the_surface():
    assert at(sphere(5), (0, 0, 0), (5, 0, 0), (8, 0, 0)) == pytest.approx([-5, 0, 3])
    assert at(box((10, 4, 2)), (5, 0, 0), (0, 0, 0))[0] == pytest.approx(0)
    assert at(capsule((0, 0, 0), (10, 0, 0), 2), (5, 2, 0), (12, 0, 0)) == pytest.approx([0, 0])
    assert at(ellipsoid((10, 5, 3)), (0, 0, 0))[0] < 0  # the centre is inside (the formula alone gives 0)
    assert at(sphere(1).translate((10, 0, 0)), (10, 0, 0))[0] == pytest.approx(-1)


def test_smooth_union_fills_the_corner_between_shapes():
    a, b = sphere(5), sphere(5).translate((8, 0, 0))
    gap = (4, 4.5, 0)  # outside both spheres, in the waist between them
    assert at(a | b, gap)[0] > 0
    assert at(a | b.k(4), gap)[0] < at(a | b, gap)[0]
    assert at(union(a, b, k=4), gap)[0] == pytest.approx(at(a | b.k(4), gap)[0])
    assert at(a - b, (6, 0, 0))[0] > 0 and at(a & b, (4, 0, 0))[0] < 0


def test_bounds_follow_transforms():
    s = capsule((0, 0, 0), (0, 0, 10), 1).rotate(math.pi / 2, Y)  # now along X
    lo, hi = s.bounds()
    assert lo == pytest.approx([-1, -1, -1]) and hi == pytest.approx([11, 1, 1], abs=1e-9)
    lo, hi = sphere(2, center=(5, 0, 0)).mirror(X).bounds()
    assert lo[0] == pytest.approx(-7) and hi[0] == pytest.approx(-3)
    assert box(4).orient(X).bounds()[1] == pytest.approx([2, 2, 2])
    with pytest.raises(ValueError, match="positive"):
        sphere(1).scale(-1)


# ---------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------
def test_mesh_is_one_watertight_body_of_the_right_size():
    m = sdf.to_mesh(sphere(10))
    assert m.is_watertight and len(m.split(only_watertight=False)) == 1
    assert m.volume == pytest.approx(4 / 3 * math.pi * 1000, rel=0.01)
    assert m.extents == pytest.approx([20, 20, 20], abs=0.1)


def test_mesh_samples_its_grid_not_point_by_point(monkeypatch):
    """level_set's sample points must fall on the grid evaluated with numpy (else meshing is ~10x slower)."""
    calls = []
    shape = (capsule((5, 5, 0), (20, 10, 0), 3) | sphere(6)).mirror(Y) | sphere(6)  # bounds not whole edges
    f = shape.f
    shape.f = lambda p: calls.append(len(p)) or f(p)
    sdf.to_mesh(shape)
    assert sum(1 for n in calls if n == 1) < 0.01 * len(calls)


def test_empty_shape_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        sdf.to_mesh(sphere(5) - sphere(6))
    with pytest.raises(ValueError, match="empty"):
        (sphere(5) - sphere(6)).on_ground()


def test_lowest_z_is_measured_not_read_from_the_bounds():
    rod = capsule((0, 0, 0), (30, 0, 0), 5).rotate(math.radians(40), Y)  # its box reaches lower than it does
    true_low = -30 * math.sin(math.radians(40)) - 5
    assert rod.bounds()[0][2] < true_low - 1
    assert sdf.lowest_z(rod) == pytest.approx(true_low, abs=0.05)


@pytest.mark.parametrize("lift", [0, -30, 25])
def test_on_ground_keeps_the_whole_object_wherever_it_was_built(lift):
    """#47: a ball built around the origin (or anywhere) keeps all but a 1.5 mm cap, standing on z = 0."""
    m = sdf.to_mesh(sphere(20).translate((0, 0, lift)).on_ground())
    assert m.is_watertight and m.bounds[0][2] == pytest.approx(0, abs=0.05)
    assert m.extents[2] == pytest.approx(40 - 1.5, abs=0.1)
    assert m.volume == pytest.approx(4 / 3 * math.pi * 20 ** 3 - math.pi * 1.5 ** 2 * (60 - 1.5) / 3, rel=0.01)
    assert sdf.to_mesh(sphere(20).on_ground(sink=4)).extents[2] == pytest.approx(36, abs=0.1)


# ---------------------------------------------------------------------------
# Sandbox and generation
# ---------------------------------------------------------------------------
def test_sandbox_runs_sdf_scripts():
    from stl_agent.sandbox import run_cadquery, validate_mesh

    ok = run_cadquery("from sdf import *\nshow_object(sphere(10) | capsule((0,0,0), (20,0,0), 4).k(3))\n",
                      language="sdf")
    assert ok.success and validate_mesh(ok.stl_path).bbox_mm == pytest.approx([34, 20, 20], abs=0.1)
    typo = run_cadquery("show_object(spheer(10))\n", language="sdf")
    assert not typo.success and "line 1" in typo.traceback and "spheer" in typo.traceback
    wrong = run_cadquery("import cadquery as cq\nshow_object(cq.Workplane().box(1, 1, 1))\n", language="sdf")
    assert not wrong.success and "sdf shape" in wrong.error_msg


def test_generate_cad_uses_the_sdf_prompt_and_runner(tmp_path):
    from stl_agent.config import SDF_EXAMPLES_PATH
    from stl_agent.generate import SYSTEM_PROMPT_SDF, generate_cad
    from stl_agent.llm import ScriptedLLM
    from stl_agent.retrieval import make_retriever

    llm = ScriptedLLM(["PLAN:\n- a ball\nBBOX: 20, 20, 20\n```python\nfrom sdf import *\nshow_object(sphere(10))\n```"])
    systems = []
    complete = llm.complete
    llm.complete = lambda system, messages: systems.append(system) or complete(system, messages)
    log = []
    res = generate_cad("a ball", llm, make_retriever(path=SDF_EXAMPLES_PATH), out_dir=tmp_path, language="sdf",
                       log=log.append)
    assert res.success and systems == [SYSTEM_PROMPT_SDF]
    assert any("SDF script" in line for line in log) and not any("CadQuery" in line for line in log)
    assert all(i.startswith("sdf_") for i in res.retrieved_ids)


# ---------------------------------------------------------------------------
# Example library
# ---------------------------------------------------------------------------
def test_example_library_matches_its_script_and_every_example_builds():
    from scripts.make_sdf_examples import EX
    from stl_agent.config import SDF_EXAMPLES_PATH

    saved = json.loads(SDF_EXAMPLES_PATH.read_text())
    assert [(e["id"], e["code"]) for e in saved] == [(e["id"], e["code"]) for e in EX], \
        "sdf_examples.json is out of date: run `pixi run -e cap6938 sdf-examples`"
    for e in saved:  # in-process (no sandbox) to keep the test quick
        shown = []
        exec(e["code"].replace("from sdf import *", "from stl_agent.sdf import *"), {"show_object": shown.append})
        m = sdf.to_mesh(shown[-1])  # the app's resolution: thin parts (the snowman's nose) open up when coarser
        assert m.is_watertight and len(m.split(only_watertight=False)) == 1, e["id"]
        assert m.bounds[0][2] == pytest.approx(0, abs=0.5), e["id"]  # stands on z = 0
        assert sorted(m.extents) == pytest.approx(sorted(e["expected_bbox_mm"]), abs=0.6), e["id"]


def test_examples_are_not_benchmark_cases():
    from stl_agent.config import SDF_EXAMPLES_PATH

    cases = Path(__file__).parents[1] / "eval" / "organic_cases.json"
    if not cases.exists():
        pytest.skip("no organic benchmark cases yet")
    subjects = {w for c in json.loads(cases.read_text())["cases"] for w in c["subject"].split()}
    for e in json.loads(SDF_EXAMPLES_PATH.read_text()):
        assert not subjects & set(e["id"].removeprefix("sdf_").split("_")), e["id"]
        assert not subjects & set(e["request"].lower().replace(",", " ").replace(".", " ").split()), e["id"]


# ---------------------------------------------------------------------------
# Routing (organic_builder setting)
# ---------------------------------------------------------------------------
HORSE = {"scope": "out_of_scope", "scope_reason": "organic", "spec": "horse", "name": "horse-statue",
         "search": "horse statue", "organic_spec": "standing horse 60 mm tall, head raised"}
LOGO = {"scope": "out_of_scope", "scope_reason": "a logo", "spec": "logo", "search": "logo", "organic_spec": ""}
COMPOSITE = {"scope": "composite", "spec": "cat on a box", "pieces": [
    {"id": "box", "source": "cadquery", "spec": "40 mm cube"},
    {"id": "cat", "source": "thingiverse", "search": ["cat"], "height_mm": 30, "fallback_spec": "stylized cat",
     "organic_spec": "sitting cat 30 mm tall", "place": {"on": "box", "relation": "on_top"}}]}


@pytest.fixture
def sculpt():
    from stl_agent.config import set_setting

    set_setting("organic_builder", "sdf")


def test_gate_keeps_organic_specs():
    from stl_agent.gate import OUT_OF_SCOPE, decide_from_report

    d = decide_from_report(HORSE, "x", 0)
    assert d.action == OUT_OF_SCOPE and d.organic_spec.startswith("standing horse")
    assert decide_from_report(COMPOSITE, "x", 0).pieces[1]["organic_spec"] == "sitting cat 30 mm tall"
    # One usable piece is not a combination: the downloaded piece's description carries over.
    single = decide_from_report({**COMPOSITE, "pieces": COMPOSITE["pieces"][1:]}, "x", 0)
    assert single.action == OUT_OF_SCOPE and single.organic_spec == "sitting cat 30 mm tall"


def test_default_setting_leaves_organic_objects_to_thingiverse():
    from stl_agent.gate import OUT_OF_SCOPE, decide_from_report
    from stl_agent.pipeline import sculpt_organic

    assert sculpt_organic(decide_from_report(HORSE, "x", 0)).action == OUT_OF_SCOPE
    assert sculpt_organic(decide_from_report(COMPOSITE, "x", 0)).pieces[1]["source"] == "thingiverse"


def test_sdf_setting_sculpts_organic_objects_and_pieces(sculpt):
    from stl_agent.gate import OUT_OF_SCOPE, PROCEED, decide_from_report
    from stl_agent.pipeline import sculpt_organic

    d = sculpt_organic(decide_from_report(HORSE, "x", 0))
    assert (d.action, d.scope, d.spec) == (PROCEED, "organic", HORSE["organic_spec"])
    assert sculpt_organic(decide_from_report(LOGO, "x", 0)).action == OUT_OF_SCOPE  # nothing to sculpt
    cat = sculpt_organic(decide_from_report(COMPOSITE, "x", 0)).pieces[1]
    assert cat["source"] == "sdf" and cat["spec"] == "sitting cat 30 mm tall" and cat["search"] == ["cat"]


def test_sculpted_pieces_need_no_token(sculpt, monkeypatch):
    from stl_agent import pipeline
    from stl_agent.gate import PROCEED, decide_from_report
    from stl_agent.sources import thingiverse

    monkeypatch.delenv(thingiverse.TOKEN_VAR, raising=False)
    monkeypatch.setattr("stl_agent.config.load_dotenv", lambda *a: None)  # ignore a token in the real .env
    d = pipeline.without_downloads_unless_token(pipeline.sculpt_organic(decide_from_report(COMPOSITE, "x", 0)))
    assert d.action == PROCEED and [p["source"] for p in d.pieces] == ["cadquery", "sdf"]


# ---------------------------------------------------------------------------
# Building: run_assembly and the app
# ---------------------------------------------------------------------------
@pytest.fixture
def assembly(monkeypatch, tmp_path):
    from stl_agent import pipeline
    from stl_agent.generate import GenResult
    from stl_agent.sources import thingiverse as tv

    calls = {"gen": [], "searched": [], "fail_sdf": False}

    def fake_generate(request, log, language="cadquery", **kw):
        calls["gen"].append((request, language))
        if language == "sdf" and calls["fail_sdf"]:
            return GenResult(success=False, error="No valid result after 4 attempt(s).")
        path = tmp_path / f"gen{len(calls['gen'])}.stl"
        trimesh.creation.box((20, 20, 10)).export(path)
        return GenResult(success=True, stl_path=str(path), code=f"# {language} code", attempts=1)

    def fake_download(thing_id, dest, client=None):
        trimesh.creation.box((5, 5, 100)).export(dest)
        return dest, []

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_generate", fake_generate)
    monkeypatch.setattr(tv, "search", lambda phrase, **kw: calls["searched"].append(phrase) or [_cand(3)])
    monkeypatch.setattr(tv, "download_stl", fake_download)
    plan = [{"id": "box", "source": "cadquery", "spec": "40 mm cube", "height_mm": None},
            {"id": "cat", "source": "sdf", "spec": "sitting cat", "search": ["cat"], "height_mm": 30,
             "fallback_spec": "stylized cat", "place": {"on": "box", "relation": "on_top", "side": "right",
                                                        "align": "center"}}]
    return dict(pipeline=pipeline, calls=calls, plan=plan, tv=tv)


def _cand(i):
    from stl_agent.sources.thingiverse import Candidate
    return Candidate(id=i, name=f"cat {i}", url=f"https://www.thingiverse.com/thing:{i}", thumbnail=f"https://cdn.example/{i}.jpg",
                     author="maker", license="CC-BY", allows_derivatives=True)


def test_sculpted_piece_is_built_and_scaled(assembly):
    stl, code, report = assembly["pipeline"].run_assembly(assembly["plan"], lambda s: None)
    assert assembly["calls"]["gen"] == [("sitting cat", "sdf"), ("40 mm cube", "cadquery")]
    assert report["pieces"][1]["source"] == "sdf" and "sculpted" in code
    from stl_agent.sources.assembly import OVERLAP

    assert trimesh.load(stl).extents[2] == pytest.approx(10 + 30 - OVERLAP, abs=0.01)  # scaled to 30 mm
    assert assembly["calls"]["searched"] == []


def test_failed_sculpted_piece_falls_back_to_a_download_then_its_stand_in(assembly, monkeypatch):
    assembly["calls"]["fail_sdf"] = True
    monkeypatch.setattr(assembly["tv"], "has_token", lambda: True)
    _, _, report = assembly["pipeline"].run_assembly(assembly["plan"], lambda s: None)
    assert assembly["calls"]["searched"] == ["cat"] and "cat 3" in report["pieces"][1]["model"]

    monkeypatch.setattr(assembly["tv"], "has_token", lambda: False)
    assembly["calls"]["gen"].clear()
    _, _, report = assembly["pipeline"].run_assembly(assembly["plan"], lambda s: None)
    assert ("stylized cat", "cadquery") in assembly["calls"]["gen"]
    assert report["pieces"][1]["source"] == "cadquery (sculpting failed)"


def test_app_sculpts_an_organic_object_and_falls_back_to_thingiverse(sculpt, monkeypatch, tmp_path):
    from stl_agent import app
    from stl_agent.gate import decide_from_report
    from stl_agent.pipeline import sculpt_organic
    from stl_agent.sources import thingiverse

    builds = []

    def fake_pipeline(spec, log, name=None, language="cadquery", **kw):
        builds.append(language)
        if len(builds) > 1:
            raise RuntimeError("the model declined")
        return str(tmp_path / f"{name}.stl"), "code", {"success": True}

    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: sculpt_organic(decide_from_report(HORSE, "x", rounds)))
    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(thingiverse, "search", lambda term, **kw: [_cand(1)])

    state = app.new_state()
    _, chat, state, build, _ = list(app.on_message("a horse", [], state))[-1]
    assert "sculpted" in chat[-1]["content"] and state["language"] == "sdf" and build.interactive
    chat, state, _, _, _ = list(app.on_build(chat, state))[-1]
    assert builds == ["sdf"] and state["result"]["stl"].endswith("horse-statue.stl")

    state.pop("result")
    chat = list(app.on_message("a horse again", chat, state))[-1][1]
    chat, state, _, log, gallery = list(app.on_build(chat, state))[-1]
    assert builds == ["sdf", "sdf"] and gallery.visible and [c.id for c in state["candidates"]] == [1]
    assert "Thingiverse" in chat[-1]["content"] and "Sculpting failed" in log


def test_code_panel_names_the_kind_of_code(monkeypatch):
    from stl_agent import app

    assert app.code_label({"language": "sdf"}) == "SDF script (sculpted)"
    assert app.code_label({"language": "cadquery"}) == app.code_label({}) == "CadQuery"
    assert app.code_label({"pieces": [{}]}).startswith("Code for each") and "downloaded" in app.code_label({"source": {}})
    state = {"result": {"stl": "x.stl", "code": "from sdf import *", "report": {"language": "sdf"}}}
    monkeypatch.setattr(app, "make_preview", lambda stl: stl)
    _, _, _, code, _ = app.show_result(state)
    assert code.label == "SDF script (sculpted)" and code.value == "from sdf import *"
