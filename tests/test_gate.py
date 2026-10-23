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
