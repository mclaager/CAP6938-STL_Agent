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
    from stl_agent.pipeline import generate_cad
    from stl_agent.retrieval import TfidfRetriever

    good = "PLAN:\n- x\nBBOX: 1, 1, 1\n```python\nimport cadquery as cq\nshow_object(cq.Workplane().box(1, 1, 1))\n```"
    r = TfidfRetriever.from_file()
    res = generate_cad("a hex nut", ScriptedLLM([good]), r, out_dir=tmp_path / "a", example_ids=["gear_001", "nope"])
    assert res.retrieved_ids == ["gear_001"]
    res = generate_cad("a hex nut", ScriptedLLM([good]), r, out_dir=tmp_path / "b", example_ids=[])
    assert res.retrieved_ids[0] == "polygon_001"  # no picks: retrieval on the request
