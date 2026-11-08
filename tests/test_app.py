"""Chat flow through the app's handlers with the gate and build stubbed (no LLM calls)."""

import pytest

from stl_agent import app
from stl_agent.gate import decide_from_report

CRITICAL = {"scope": "in_scope", "spec": "stand",
            "unknowns": [{"param": "width", "default": "75", "critical": True, "question": "Phone width?"}]}
READY = {"scope": "in_scope", "spec": "stand 80mm", "name": "phone-stand",
         "unknowns": [{"param": "angle", "default": "60", "critical": False}]}


def last(gen):
    *_, final = gen
    return final


def test_clarify_then_build(monkeypatch, tmp_path):
    calls = []

    def fake_gate(conversation, rounds):
        calls.append(rounds)
        return decide_from_report(CRITICAL if rounds == 0 else READY, "x", rounds)

    def fake_pipeline(spec, log, name=None, **kw):
        log("working on " + spec)
        return str(tmp_path / f"{name}.stl"), "code", {"success": True}

    monkeypatch.setattr(app, "run_gate", fake_gate)
    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    state = app.new_state()

    chat = last(app.on_message("phone stand", [], state))[1]
    assert "Phone width?" in chat[-1]["content"]
    assert state["rounds"] == 1 and state["spec"] is None

    chat = last(app.on_message("80mm", chat, state))[1]
    assert "stand 80mm" in chat[-1]["content"]
    assert state["spec"] == "stand 80mm" and state["name"] == "phone-stand"

    chat, state, _, log, _ = last(app.on_build(chat, state))
    assert "phone-stand.stl" in chat[-1]["content"]
    assert "working on stand 80mm" in log
    assert state["result"]["stl"].endswith("phone-stand.stl")
    assert state["spec"] is None and state["rounds"] == 0
    assert calls == [0, 1]


# ---------------------------------------------------------------------------
# Thingiverse fallback for out-of-scope requests (#9)
# ---------------------------------------------------------------------------
HORSE = {"scope": "out_of_scope", "scope_reason": "organic shape", "spec": "horse", "name": "horse-statue",
         "search": "horse statue"}


def candidate(i):
    from stl_agent.sources.thingiverse import Candidate
    return Candidate(id=i, name=f"Horse {i}", url=f"https://www.thingiverse.com/thing:{i}",
                     thumbnail=f"https://cdn.example/{i}.jpg", author="maker", license="CC-BY")


def test_out_of_scope_shows_thingiverse_results(monkeypatch):
    from stl_agent.sources import thingiverse

    terms = []
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    monkeypatch.setattr(thingiverse, "search", lambda term, **kw: terms.append(term) or [candidate(1), candidate(2)])
    state = app.new_state()
    _, chat, state, build, gallery = last(app.on_message("a statue of a horse", [], state))
    assert terms == ["horse statue"]
    assert [c.id for c in state["candidates"]] == [1, 2]
    assert gallery.visible and "Thingiverse" in chat[-1]["content"]
    assert state["spec"] is None


def test_out_of_scope_auto_downloads_without_gallery(monkeypatch, tmp_path):
    from stl_agent.config import reload_settings, set_setting
    from stl_agent.sources import thingiverse

    fetched = []

    def fake_fetch(c, log, name=None):
        fetched.append(c.id)
        if c.id == 1:
            raise thingiverse.ThingiverseError("no STL")
        return str(tmp_path / "horse-statue.stl"), {"success": True, "source": {"other_stl_files": []}}

    set_setting("thingiverse_pick", "auto")
    reload_settings()
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    monkeypatch.setattr(thingiverse, "search", lambda term, **kw: [candidate(1), candidate(2)])
    monkeypatch.setattr(app, "run_fetch", fake_fetch)
    state = app.new_state()
    _, chat, state, _, gallery = last(app.on_message("a statue of a horse", [], state))
    assert fetched == [1, 2] and not gallery.visible  # first one failed, second downloaded
    assert "Horse 2" in chat[-1]["content"] and state["result"]["stl"].endswith("horse-statue.stl")


def test_auto_pick_without_a_matching_result_shows_the_gallery(monkeypatch):
    """#39: auto mode downloads only a result named or tagged as the object; if none is, the user picks."""
    from stl_agent.config import reload_settings, set_setting
    from stl_agent.sources import thingiverse

    calls = []

    def search(term, words=None, strict=False, **kw):
        calls.append((term, words, strict))
        return [] if strict else [candidate(1)]

    set_setting("thingiverse_pick", "auto")
    reload_settings()
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    monkeypatch.setattr(thingiverse, "search", search)
    monkeypatch.setattr(app, "run_fetch", lambda *a, **k: pytest.fail("downloaded a result that isn't a horse"))
    state = app.new_state()
    _, chat, state, _, gallery = last(app.on_message("a statue of a horse", [], state))
    assert calls == [("horse statue", {"horse"}, True), ("horse statue", {"horse"}, False)]
    assert gallery.visible and [c.id for c in state["candidates"]] == [1]
    assert 'named or tagged "horse"' in chat[-1]["content"]


def test_out_of_scope_without_token_declines_as_before(monkeypatch):
    from stl_agent.sources import thingiverse

    monkeypatch.delenv(thingiverse.TOKEN_VAR, raising=False)
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    state = app.new_state()
    _, chat, state, _, gallery = last(app.on_message("a statue of a horse", [], state))
    d = decide_from_report(HORSE, "x", 0)
    assert chat[-1]["content"] == app.format_decision(d)  # the plain decline, nothing about Thingiverse
    assert not gallery.visible and state["candidates"] == []


def test_pick_downloads_and_credits(monkeypatch, tmp_path):
    def fake_fetch(c, log, name=None):
        log("Saved x")
        return str(tmp_path / "horse-statue.stl"), {"success": True, "source": {"other_stl_files": ["base.stl"]}}

    monkeypatch.setattr(app, "run_fetch", fake_fetch)
    state = app.new_state()
    state["candidates"], state["name"] = [candidate(1), candidate(2)], "horse-statue"

    class Evt:  # stands in for gr.SelectData
        index = 1

    chat, state, log, gallery = app.on_pick(Evt(), [], state)
    assert not gallery.visible and state["candidates"] == []  # picked: the gallery goes away
    msg = chat[-1]["content"]
    assert "Horse 2" in msg and "maker" in msg and "CC-BY" in msg and "base.stl" in msg
    assert state["result"]["stl"].endswith("horse-statue.stl")
    assert "thing:2" in state["result"]["code"]
    assert "Saved x" in log


# ---------------------------------------------------------------------------
# Combined parts (#10)
# ---------------------------------------------------------------------------
COMPOSITE = {"scope": "composite", "spec": "a cat on a tree", "name": "cat-on-tree", "pieces": [
    {"id": "tree", "source": "thingiverse", "search": ["oak tree", "tree"], "height_mm": 120, "fallback_spec": "a tree"},
    {"id": "cat", "source": "thingiverse", "search": ["cat"], "height_mm": 30, "fallback_spec": "a cat",
     "place": {"on": "tree", "relation": "on_top"}},
    {"id": "plate", "source": "cadquery", "spec": "name plate", "place": {"on": "tree", "relation": "beside"}},
]}


class Evt:  # stands in for gr.SelectData
    def __init__(self, index):
        self.index = index


@pytest.fixture
def composite(monkeypatch, tmp_path):
    from stl_agent.sources import thingiverse

    calls = []

    def fake_assembly(pieces, log, name=None, picks=None):
        calls.append({"pieces": pieces, "picks": picks, "name": name})
        log("assembling")
        return (str(tmp_path / "cat-on-tree.stl"), "code",
                {"success": True, "pieces": [{"id": p["id"], "source": p["source"]} for p in pieces]})

    results = {"oak tree": [], "tree": [candidate(1), candidate(2)], "cat": [candidate(3)]}
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(thingiverse, "search", lambda phrase, **kw: list(results.get(phrase, [])))
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(COMPOSITE, "x", rounds))
    monkeypatch.setattr(app, "run_assembly", fake_assembly)
    state = app.new_state()
    chat = last(app.on_message("a cat on a tree with a name plate", [], state))[1]
    return dict(state=state, chat=chat, calls=calls, results=results)


def test_composite_plan_is_shown(composite):
    msg = composite["chat"][-1]["content"]
    assert "**tree** (from Thingiverse, 120 mm tall)" in msg and "on top of tree" in msg
    assert "**plate** (built with CadQuery)" in msg and "right of tree" in msg
    assert len(composite["state"]["pieces"]) == 3


def test_composite_manual_picks_then_assembles(composite):
    state, chat = composite["state"], composite["chat"]
    chat, state, _, _, gallery = last(app.on_build(chat, state))
    assert gallery.visible and state["picking"] == "tree" and "Pick the **tree**" in chat[-1]["content"]
    assert composite["calls"] == []

    chat, state, _, gallery = app.on_pick(Evt(1), chat, state)
    assert state["picking"] == "cat" and state["picks"]["tree"].id == 2 and gallery.visible
    assert not state["assemble"] and len(list(app.on_assemble(chat, state))) == 1  # still picking: no build
    assert composite["calls"] == []

    chat, state, _, gallery = app.on_pick(Evt(0), chat, state)  # candidate 3 for the cat
    assert state["assemble"] and not gallery.visible

    chat, state, _, log = last(app.on_assemble(chat, state))
    call = composite["calls"][0]
    assert call["picks"]["tree"].id == 2 and call["picks"]["cat"].id == 3 and call["name"] == "cat-on-tree"
    assert "assembling" in log and "Pieces:" in chat[-1]["content"]
    assert state["result"]["stl"].endswith("cat-on-tree.stl") and state["pieces"] == []


def test_composite_piece_without_results_gets_fallback(composite):
    composite["results"].update(tree=[], cat=[])
    chat, state, _, _, gallery = last(app.on_build(composite["chat"], composite["state"]))
    assert composite["calls"][0]["picks"] == {"tree": None, "cat": None}
    assert "No Thingiverse model found for the tree" in " ".join(m["content"] for m in chat)


def test_composite_auto_skips_gallery(composite):
    from stl_agent.config import set_setting, reload_settings

    set_setting("thingiverse_pick", "auto")
    reload_settings()
    chat, state, _, _, gallery = last(app.on_build(composite["chat"], composite["state"]))
    assert composite["calls"][0]["picks"] is None and not gallery.visible


def test_composite_needs_token(composite, monkeypatch):
    from stl_agent.sources import thingiverse

    monkeypatch.delenv(thingiverse.TOKEN_VAR)
    chat, *_ = last(app.on_build(composite["chat"], composite["state"]))
    assert "THINGIVERSE_TOKEN" in chat[-1]["content"] and composite["calls"] == []


def test_gallery_shows_fetched_thumbnails_or_links(monkeypatch):
    import io

    from PIL import Image

    from stl_agent.sources import thingiverse

    png = io.BytesIO()
    Image.new("RGB", (1024, 1024), "red").save(png, format="PNG")
    monkeypatch.setattr(thingiverse, "thumbnails", lambda urls: [png.getvalue(), None])
    value = app.gallery_update([candidate(1), candidate(2)]).value
    # Gradio's postprocessed gallery: the fetched image was saved locally, the other is still the link.
    first, second = (item["image"] for item in value)
    assert first["url"].startswith("/gradio_api/file=")
    assert second["url"] == "https://cdn.example/2.jpg"

