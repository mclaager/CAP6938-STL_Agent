"""Combined parts (#10): placing and merging pieces, and run_assembly with generation and downloads stubbed."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from stl_agent.sources import assembly as A


def base_and(piece, **where):
    """Plan for a base box plus one piece placed on it."""
    return [{"id": "base"}, {"id": "top", "place": {"on": "base", **where}}]


def test_scale_to_height():
    m = A.scale_to_height(trimesh.creation.box((10, 20, 40)), 10)
    assert np.allclose(sorted(m.extents), [2.5, 5, 10])


def test_surface_heights_like_rays_down():
    cone = trimesh.creation.cone(radius=20, height=40)
    z = A.surface_heights(cone, np.array([[0, 0], [10, 0], [30, 0]]))
    assert np.allclose(z[:2], [40, 20]) and np.isnan(z[2])


def test_on_top_rests_on_the_surface_not_the_bounding_box():
    # A ball on a cone's tip at x=0 sits higher than a ball beside the tip would: it lands on the surface.
    cone, ball = trimesh.creation.cone(radius=20, height=40), trimesh.creation.icosphere(radius=5)
    placed = A.place({"base": cone, "top": ball}, base_and(ball, relation="on_top"))
    assert placed["top"].bounds[0, 2] == pytest.approx(40 - A.OVERLAP, abs=0.05)
    # Off-centre on a step: the piece lands on the lower step under it, not on the top of the bounding box.
    step = trimesh.util.concatenate([trimesh.creation.box((40, 20, 10), transform=trimesh.transformations.translation_matrix((0, 0, 5))),
                                     trimesh.creation.box((20, 20, 20), transform=trimesh.transformations.translation_matrix((-10, 0, 10)))])
    small = trimesh.creation.box((8, 8, 8))
    placed = A.place({"base": step, "top": small}, base_and(small, relation="on_top", align="right"))
    assert placed["top"].bounds[0, 2] == pytest.approx(10 - A.OVERLAP, abs=0.05)
    assert placed["top"].bounds[1, 0] == pytest.approx(placed["base"].bounds[1, 0])  # flush with the right edge


def test_beside_and_under():
    a, b = trimesh.creation.box((10, 10, 10)), trimesh.creation.box((4, 4, 4))
    placed = A.place({"base": a, "top": b}, base_and(b, relation="beside", side="front"))
    assert placed["top"].bounds[1, 1] == pytest.approx(placed["base"].bounds[0, 1] + A.OVERLAP)  # front is -Y
    assert placed["top"].bounds[0, 2] == pytest.approx(placed["base"].bounds[0, 2])
    placed = A.place({"base": a, "top": b}, base_and(b, relation="under"))
    assert placed["top"].bounds[1, 2] == pytest.approx(placed["base"].bounds[0, 2] + A.OVERLAP)
    assert min(m.bounds[0, 2] for m in placed.values()) == pytest.approx(0)  # whole assembly on Z = 0


def test_inside_lands_on_the_inner_floor():
    cup = trimesh.boolean.difference([trimesh.creation.box((30, 30, 20)),
                                      trimesh.creation.box((26, 26, 20), transform=trimesh.transformations.translation_matrix((0, 0, 2)))],
                                     engine="manifold")
    ball = trimesh.creation.icosphere(radius=4)
    placed = A.place({"base": cup, "top": ball}, base_and(ball, relation="inside"))
    assert placed["top"].bounds[0, 2] == pytest.approx(2 - A.OVERLAP, abs=0.05)


def test_overlapping_pieces_union_into_one_body():
    a, b = trimesh.creation.box((10, 10, 10)), trimesh.creation.icosphere(radius=4)
    placed = A.place({"base": a, "top": b}, base_and(b, relation="on_top"))
    merged, notes = A.merge(list(placed.values()), ["base", "top"])
    assert merged.is_watertight and len(merged.split()) == 1 and notes == []


def test_open_mesh_is_repaired_or_joined_with_a_note():
    box = trimesh.creation.box((10, 10, 10))
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-2])  # one side missing
    merged, notes = A.merge([trimesh.creation.box((20, 20, 2)), open_box], ["plate", "box"])
    assert notes and "box" in notes[0]
    assert merged.is_watertight or any("overlapping bodies" in n for n in notes)


def test_non_repairable_mesh_is_joined():
    sheet = trimesh.Trimesh([[0, 0, 0], [10, 0, 0], [0, 10, 0]], [[0, 1, 2]])  # a single triangle
    merged, notes = A.merge([trimesh.creation.box((5, 5, 5)), sheet], ["box", "sheet"])
    assert any("overlapping bodies" in n for n in notes)
    assert len(merged.faces) == 12 + 1


# ---------------------------------------------------------------------------
# run_assembly: Thingiverse pieces first, auto/manual picks, fallback pieces
# ---------------------------------------------------------------------------
class FakeGen:
    def __init__(self, folder: Path, size=(20, 20, 10)):
        self.folder, self.size, self.calls, self.success = folder, size, [], True

    def __call__(self, request, log, **kw):
        from stl_agent.generate import GenResult

        self.calls.append(request)
        path = self.folder / f"gen{len(self.calls)}.stl"
        trimesh.creation.box(self.size).export(path)
        res = GenResult(success=self.success, stl_path=str(path), code=f"# code for {request}")
        res.error = "" if self.success else "Best-effort result: mesh did not pass all checks after all attempts."
        res.attempts = 1
        return res


def cand(i, derivatives=True):
    from stl_agent.sources.thingiverse import Candidate
    return Candidate(id=i, name=f"model {i}", url=f"https://www.thingiverse.com/thing:{i}", thumbnail="",
                     author="maker", license="CC-BY", allows_derivatives=derivatives)


PLAN = [
    {"id": "plinth", "source": "cadquery", "spec": "round plinth", "height_mm": None},
    {"id": "horse", "source": "thingiverse", "search": ["rearing horse", "horse"], "height_mm": 50,
     "fallback_spec": "stylized horse", "place": {"on": "plinth", "relation": "on_top", "side": "right", "align": "center"}},
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    from stl_agent import pipeline
    from stl_agent.sources import thingiverse as tv

    order, searched = [], []
    gen = FakeGen(tmp_path)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_generate", lambda req, log, **kw: order.append("gen") or gen(req, log, **kw))

    results = {"rearing horse": [], "horse": [cand(1, derivatives=False), cand(2)]}
    monkeypatch.setattr(tv, "search", lambda phrase, **kw: searched.append(phrase) or results.get(phrase, []))

    def fake_download(thing_id, dest, client=None):
        order.append(f"download {thing_id}")
        trimesh.creation.box((5, 5, 100)).export(dest)
        return dest, []

    monkeypatch.setattr(tv, "download_stl", fake_download)
    return dict(pipeline=pipeline, order=order, searched=searched, gen=gen, results=results, tmp=tmp_path)


def test_auto_pick_tries_phrases_and_skips_no_derivatives(env):
    log = []
    stl, code, report = env["pipeline"].run_assembly(PLAN, log.append, name="horse-on-plinth")
    assert env["searched"] == ["rearing horse", "horse"]
    assert env["order"] == ["download 2", "gen"]  # downloads before any generation request
    merged = trimesh.load(stl)
    assert merged.extents[2] == pytest.approx(10 + 50 - A.OVERLAP, abs=0.01)  # horse scaled to 50 mm, on the plinth
    assert report["success"] and [r["id"] for r in report["pieces"]] == ["plinth", "horse"]
    assert "model 2 by maker" in report["pieces"][1]["model"] and "model 2 by maker" in code
    credits = json.loads(Path(stl).with_suffix(".source.json").read_text())
    assert credits["pieces"][0]["piece"] == "horse" and credits["pieces"][0]["id"] == 2


def test_auto_pick_skips_results_that_are_not_the_piece(env, monkeypatch):
    """#39: a search that only finds other things (the cat placed on a Bulbasaur) moves on to the next phrase."""
    from stl_agent.sources import thingiverse as tv

    pool = {"rearing horse": [tv.Candidate(id=8, name="Rearing unicorn", url="u", thumbnail="", author="a",
                                           license="CC-BY", allows_derivatives=True)],
            "horse": [tv.Candidate(id=9, name="Pony", url="u", thumbnail="", author="a", license="CC-BY",
                                   allows_derivatives=True, tags=["horse", "pony"])]}
    calls = []

    def search(phrase, words=None, strict=False, **kw):
        calls.append((phrase, words, strict))
        return [c for c in pool[phrase] if not strict or tv.match(c, words)]

    monkeypatch.setattr(tv, "search", search)
    log = []
    env["pipeline"].run_assembly(PLAN, log.append)
    assert calls == [("rearing horse", {"horse"}, True), ("horse", {"horse"}, True)]
    assert env["order"] == ["download 9", "gen"]  # the pony tagged "horse", not the unicorn
    assert any("no result for 'rearing horse' is named or tagged horse" in line for line in log)


def test_no_model_builds_the_fallback(env):
    env["results"]["horse"] = []
    stl, _, report = env["pipeline"].run_assembly(PLAN, lambda s: None)
    assert env["gen"].calls == ["round plinth", "stylized horse"]  # plan order
    assert "no Thingiverse model" in report["pieces"][1]["source"]
    assert not Path(stl).with_suffix(".source.json").exists()
    assert trimesh.load(stl).extents[2] == pytest.approx(10 + 50 - A.OVERLAP, abs=0.01)  # stand-in scaled too


def test_no_model_and_no_fallback_stops_before_generating(env):
    env["results"]["horse"] = []
    plan = [PLAN[0], {**PLAN[1], "fallback_spec": ""}]
    with pytest.raises(RuntimeError, match="horse"):
        env["pipeline"].run_assembly(plan, lambda s: None)
    assert env["gen"].calls == []


def test_manual_picks_are_used(env):
    stl, _, report = env["pipeline"].run_assembly(PLAN, lambda s: None, picks={"horse": cand(7)})
    assert env["searched"] == [] and env["order"] == ["download 7", "gen"]
    assert "model 7" in report["pieces"][1]["model"]
    _, _, report = env["pipeline"].run_assembly(PLAN, lambda s: None, picks={"horse": None})
    assert "no Thingiverse model" in report["pieces"][1]["source"]


def test_best_effort_piece_is_flagged(env):
    env["gen"].success = False
    _, _, report = env["pipeline"].run_assembly(PLAN, lambda s: None)
    assert "Best-effort" in report["pieces"][0]["warning"] and "warning" not in report["pieces"][1]

