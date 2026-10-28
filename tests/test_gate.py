"""Offline checks of the gate's decision rules and reply parsing (no LLM calls)."""

from stl_agent.gate import (CLARIFY, MAX_ROUNDS, OUT_OF_SCOPE, PROCEED, decide_from_report,
                            parse_gate_reply, slugify)

BASE = {"scope": "in_scope", "spec": "phone stand", "unknowns": [
    {"param": "phone width", "default": "75 mm", "critical": True, "question": "How wide is your phone?"},
    {"param": "angle", "default": "60 deg", "critical": False, "question": "What angle?"},
]}


def test_critical_unknown_asks():
    d = decide_from_report(BASE, "x", 0)
    assert d.action == CLARIFY
    assert d.questions == ["How wide is your phone?"]


def test_proceeds_after_max_rounds():
    assert decide_from_report(BASE, "x", MAX_ROUNDS).action == PROCEED


def test_non_critical_unknowns_proceed():
    assert decide_from_report({**BASE, "unknowns": BASE["unknowns"][1:]}, "x", 0).action == PROCEED


def test_out_of_scope():
    assert decide_from_report({"scope": "out_of_scope"}, "x", 0).action == OUT_OF_SCOPE


def test_empty_report_falls_back_to_raw_request():
    assert decide_from_report({}, "raw", 0).spec == "raw"


def test_names_are_slugified():
    assert decide_from_report({**BASE, "name": "Phone Stand/../x"}, "x", 0).name == "phone-stand-x"
    assert slugify("An 80mm Flange, 6 holes!") == "an-80mm-flange-6-holes"


def test_search_terms_fall_back_to_name():
    d = decide_from_report({"scope": "out_of_scope", "name": "Horse Statue", "search": "  horse   statue "}, "x", 0)
    assert d.search == "horse statue"
    assert decide_from_report({"scope": "out_of_scope", "name": "Horse Statue"}, "x", 0).search == "horse statue"


def test_parse_fenced_json():
    assert parse_gate_reply('ok\n```json\n{"scope": "in_scope"}\n```')["scope"] == "in_scope"


def test_gate_picks_are_parsed_and_made_up_ids_dropped():
    from stl_agent.gate.llm_gate import LLMGate
    from stl_agent.llm import ScriptedLLM

    library = [{"id": "gear_001", "request": "a gear"}, {"id": "text_002", "request": "engraved text"}]
    reply = ('```json\n{"scope": "in_scope", "spec": "a gear", "unknowns": [],'
             ' "examples": ["gear_001", "made_up_999", "gear_001", "text_002", "x", "y"]}\n```')
    llm = ScriptedLLM([reply])
    d = LLMGate(llm, library=library).decide([{"role": "user", "content": "a cog"}])
    assert d.example_ids == ["gear_001", "text_002"]


def test_gate_prompt_includes_library_only_when_picking():
    from stl_agent.gate.llm_gate import GATE_PROMPT, LLMGate

    assert LLMGate(None).prompt == GATE_PROMPT
    assert "gear_001: a gear" in LLMGate(None, library=[{"id": "gear_001", "request": "a gear"}]).prompt


def test_no_examples_field_means_gate_did_not_pick():
    assert decide_from_report({"spec": "x"}, "x", 0).example_ids is None
    assert decide_from_report({"spec": "x", "examples": []}, "x", 0).example_ids == []


def test_pipeline_prefers_gate_picks(tmp_path):
    from stl_agent.llm import ScriptedLLM
    from stl_agent.generate import generate_cad
    from stl_agent.retrieval import TfidfRetriever

    good = "PLAN:\n- x\nBBOX: 1, 1, 1\n```python\nimport cadquery as cq\nshow_object(cq.Workplane().box(1, 1, 1))\n```"
    r = TfidfRetriever.from_file()
    res = generate_cad("a hex nut", ScriptedLLM([good]), r, out_dir=tmp_path / "a", example_ids=["gear_001", "nope"])
    assert res.retrieved_ids == ["gear_001"]
    res = generate_cad("a hex nut", ScriptedLLM([good]), r, out_dir=tmp_path / "b", example_ids=[])
    assert res.retrieved_ids[0] == "polygon_001"  # no picks: retrieval on the request


# ---------------------------------------------------------------------------
# Composite plans (#10)
# ---------------------------------------------------------------------------
TREE = {"id": "Tree", "source": "thingiverse", "search": "oak tree", "height_mm": "120", "fallback_spec": "stylized tree"}
CAT = {"id": "cat", "source": "thingiverse", "search": ["sitting cat", "cat", "kitten", "kitty"], "height_mm": 30,
       "place": {"on": "tree", "relation": "on_top", "align": "front"}}
PLATE = {"id": "plate", "source": "cadquery", "spec": "40x12x3 mm name plate",
         "place": {"on": "tree", "relation": "beside", "side": "sideways"}}


def test_composite_plan_is_parsed():
    from stl_agent.gate.base import MAX_SEARCHES

    d = decide_from_report({"scope": "composite", "spec": "cat on a tree", "pieces": [TREE, CAT, PLATE]}, "x", 0)
    assert d.action == PROCEED and d.scope == "composite"
    tree, cat, plate = d.pieces
    assert tree["id"] == "tree" and tree["search"] == ["oak tree"] and tree["height_mm"] == 120 and "place" not in tree
    assert len(cat["search"]) == MAX_SEARCHES and cat["place"] == {"on": "tree", "relation": "on_top",
                                                                    "side": "right", "align": "front"}
    assert plate["spec"] and plate["place"]["side"] == "right"  # unknown side -> default


def test_composite_drops_bad_pieces():
    bad = [
        {**CAT, "id": "dog", "place": {"on": "moon", "relation": "on_top"}},  # target not an earlier piece
        {**CAT, "id": "bird", "place": {"on": "tree", "relation": "orbiting"}},  # unknown relation
        {**CAT, "id": "tree"},  # repeated id
        {"id": "rock", "source": "sculpt"},  # unknown source
        {"id": "box", "source": "cadquery"},  # no spec
    ]
    d = decide_from_report({"scope": "composite", "pieces": [TREE, *bad, CAT]}, "x", 0)
    assert [p["id"] for p in d.pieces] == ["tree", "cat"]
    assert "moon" in d.note and "orbiting" in d.note


def test_composite_with_one_piece_falls_back():
    d = decide_from_report({"scope": "composite", "name": "tree", "pieces": [TREE]}, "x", 0)
    assert d.action == OUT_OF_SCOPE and d.search == "oak tree" and not d.pieces
    d = decide_from_report({"scope": "composite", "pieces": [{"id": "box", "source": "cadquery", "spec": "a box"}]}, "x", 0)
    assert d.action == PROCEED and d.scope == "in_scope" and d.spec == "a box" and not d.pieces


def test_thingiverse_plan_without_token_is_declined(monkeypatch):
    from stl_agent import config, pipeline
    from stl_agent.gate.base import CLARIFY
    from stl_agent.sources import thingiverse

    monkeypatch.setattr(config, "load_dotenv", lambda: None)  # the real .env may hold a token
    monkeypatch.delenv(thingiverse.TOKEN_VAR, raising=False)
    plan = {"scope": "composite", "spec": "cat on a tree", "pieces": [TREE, CAT],
            "unknowns": [{"param": "pose", "critical": True, "question": "Which pose?"}]}
    d = pipeline.without_downloads_unless_token(decide_from_report(plan, "x", 0))
    assert d.action == OUT_OF_SCOPE and d.scope == "out_of_scope" and d.pieces == [] and d.questions == []

    geometric = {"scope": "composite", "pieces": [{"id": "a", "source": "cadquery", "spec": "box"},
                                                  {"id": "b", "source": "cadquery", "spec": "cone",
                                                   "place": {"on": "a", "relation": "on_top"}}]}
    assert pipeline.without_downloads_unless_token(decide_from_report(geometric, "x", 0)).action == PROCEED

    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    d = pipeline.without_downloads_unless_token(decide_from_report(plan, "x", 0))
    assert d.action == CLARIFY and len(d.pieces) == 2

