"""Size-check replay (eval/size_check.py, #23): finding size-only rejections and judging a replayed reply.
No LLM requests."""

import json

from eval import size_check as S
from stl_agent.generate import parse_response

BOX = "import cadquery as cq\nshow_object(cq.Workplane('XY').box(20, 10, 5))\n"
SIZE_ONLY = {"valid": False, "watertight": True, "n_components": 1, "volume_mm3": 1000.0, "bbox_ok": False,
             "bbox_mm": [20.0, 10.0, 5.0], "issues": ["Bounding box [20.0, 10.0, 5.0] mm does not match expected"]}


def rec(attempt, **kw):
    return {"attempt": attempt, "plan": "- a block", "bbox_claimed": [20, 10, 6], "code": BOX, **kw}


def test_size_only_needs_a_sound_mesh_and_a_claim():
    assert S.size_only(rec(1, validation=SIZE_ONLY))
    assert not S.size_only(rec(1, validation={**SIZE_ONLY, "watertight": False}))
    assert not S.size_only(rec(1, validation={**SIZE_ONLY, "n_components": 2}))
    assert not S.size_only(rec(1, validation={**SIZE_ONLY, "bbox_ok": None}))
    assert not S.size_only(rec(1, outcome="exec_error"))


def test_rebuilt_reply_parses_back():
    p = parse_response(S.rebuilt_reply(rec(1)))
    assert p.bbox == (20, 10, 6) and p.code.strip() == BOX.strip() and p.plan == "- a block"


def test_find_attempts_rebuilds_the_conversation(tmp_path, monkeypatch):
    run = tmp_path / "bench" / "m" / "lib-x" / "rag" / "c1" / "rep0"
    run.mkdir(parents=True)
    log = [rec(1, outcome="exec_error", error="NameError: x"), rec(2, outcome="invalid_mesh", validation=SIZE_ONLY)]
    (run / "report.json").write_text(json.dumps({"retrieved_examples": [], "attempt_log": log}))
    skipped = tmp_path / "bench" / "m" / "lib-x" / "rag-hints" / "c1" / "rep0"
    skipped.mkdir(parents=True)
    (skipped / "report.json").write_text((run / "report.json").read_text())
    monkeypatch.setattr(S, "OUTPUT_DIR", tmp_path)
    cases = {"c1": {"id": "c1", "request": "a block 20x10x5", "expected_bbox_mm": [20, 10, 5]}}

    [a] = S.find_attempts(tmp_path / "bench" / "m", cases, "lib-x")
    assert a["key"] == "m/lib-x/rag/c1/rep0#2" and a["right_size"] and a["claimed"] == [20, 10, 6]
    roles = [m["role"] for m in a["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]  # request, attempt 1, its error, attempt 2
    assert "NameError" in a["messages"][2]["content"]
    for messages in S.VARIANTS.values():
        for message in messages:
            assert "20 x 10 x 6" in message(a["val"], a["claimed"]) or "expected" in message(a["val"], a["claimed"])
    assert "20 x 10 x 5" not in S.msg_blind(a["val"], a["claimed"])  # the measured size stays hidden


def test_judge_reply_runs_the_script_against_its_own_claim():
    attempt = {"code": BOX, "expected": [20, 10, 5]}
    fixed = S.judge_reply(f"PLAN:\n- same\nBBOX: 20, 10, 5\n```python\n{BOX}```", attempt)
    assert fixed["outcome"] == "accepted_right" and fixed["code_unchanged"]
    stamped = S.judge_reply(f"BBOX: 20, 10, 5\n```python\n{BOX}```", {"code": BOX, "expected": [20, 10, 8]})
    assert stamped["outcome"] == "accepted_wrong"  # the claim now matches a part of the wrong size
    assert S.judge_reply(f"BBOX: 20, 10, 9\n```python\n{BOX}```", attempt)["outcome"] == "rejected"
    assert S.judge_reply("no code", attempt)["outcome"] == "no_code"


def test_blend_shows_the_measured_size_only_on_a_second_size_failure(tmp_path, monkeypatch):
    from stl_agent import llm as llm_mod
    from stl_agent.llm import ScriptedLLM

    run = tmp_path / "bench" / "m" / "lib-x" / "rag" / "c1" / "rep0"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"retrieved_examples": [], "attempt_log": [
        rec(1, outcome="invalid_mesh", validation=SIZE_ONLY)]}))
    monkeypatch.setattr(S, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(S, "OUT_DIR", tmp_path / "results")
    monkeypatch.setattr(S, "library_id", lambda: "lib-x")
    monkeypatch.setattr(S, "load_cases", lambda: [{"id": "c1", "request": "a block", "expected_bbox_mm": [20, 10, 5]}])
    replies = [f"BBOX: 20, 10, 7\n```python\n{BOX}```",  # blind: still the wrong claim
               f"BBOX: 20, 10, 5\n```python\n{BOX}```"]  # then corrected, after seeing the measured size
    scripted = ScriptedLLM(replies)
    sent = []
    complete = scripted.complete
    scripted.complete = lambda system, messages: sent.append(messages[-1]["content"]) or complete(system, messages)
    monkeypatch.setattr(llm_mod, "make_llm", lambda **kw: scripted)

    S.replay("m", ["blend"], budget=5, log=lambda m: None)
    [row] = S.read(tmp_path / "results" / "m.jsonl")
    assert row["outcomes"] == ["rejected", "accepted_right"] and row["requests"] == 2
    assert "20 x 10 x 5" not in sent[0] and "20 x 10 x 5" in sent[1] and "20 x 10 x 7" in sent[1]


def test_shipped_messages_match_the_tested_ones():
    from stl_agent.generate import build_size_message
    from stl_agent.sandbox import ValidationReport

    val = ValidationReport(valid=False, watertight=True, n_components=1, volume_mm3=1000.0, bbox_ok=False,
                           bbox_mm=[20.0, 10.0, 5.0])
    assert build_size_message(val, (20, 10, 6), "ask") == S.msg_blind(SIZE_ONLY, (20, 10, 6))
    assert build_size_message(val, (20, 10, 6), "measured") == S.msg_claim(SIZE_ONLY, (20, 10, 6))
    assert build_size_message(val, (20, 10, 6), "fix") is None


def test_generate_cad_sends_the_size_message_only_for_a_size_only_failure(tmp_path):
    from stl_agent.generate import generate_cad
    from stl_agent.llm import ScriptedLLM
    from stl_agent.retrieval import make_retriever

    two = "import cadquery as cq\nshow_object(cq.Workplane('XY').box(5, 5, 5).union(cq.Workplane('XY').box(5, 5, 5).translate((20, 0, 0))))\n"
    for mode, expect in (("ask", "Either the script or the BBOX line is wrong"), ("fix", "Fix the script")):
        llm = ScriptedLLM([f"BBOX: 20, 10, 9\n```python\n{BOX}```", f"BBOX: 20, 10, 5\n```python\n{BOX}```"])
        res = generate_cad("a block", llm, make_retriever(), out_dir=tmp_path / mode, size_feedback=mode, top_k=0)
        assert res.success and expect in res.history[0]["feedback"]
    # Two bodies: not a size-only failure, so the general message, whatever the setting.
    llm = ScriptedLLM([f"BBOX: 25, 5, 9\n```python\n{two}```", f"BBOX: 20, 10, 5\n```python\n{BOX}```"])
    res = generate_cad("a block", llm, make_retriever(), out_dir=tmp_path / "two", size_feedback="ask", top_k=0)
    assert "disconnected" in res.history[0]["feedback"] and "Either the script" not in res.history[0]["feedback"]
