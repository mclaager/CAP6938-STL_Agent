"""Visual judge: image messages, reply parsing, the eval CLI's bookkeeping (no real LLM calls)."""

import json

import trimesh

from stl_agent.judge import make_judge, parse_judge_reply
from stl_agent.llm import ScriptedLLM, anthropic_content, openai_content

PNG = b"\x89PNG fake"


def test_image_parts_convert_per_backend():
    content = [{"type": "text", "text": "hi"}, {"type": "image", "png": PNG}]
    a = anthropic_content(content)
    assert a[0] == {"type": "text", "text": "hi"}
    assert a[1]["type"] == "image" and a[1]["source"]["media_type"] == "image/png"
    o = openai_content(content)
    assert o[1]["type"] == "image_url" and o[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert anthropic_content("plain") == "plain" and openai_content("plain") == "plain"


def test_parse_judge_reply():
    j = parse_judge_reply('```json\n{"matches": ["disc"], "problems": ["no holes"], "score": 4}\n```')
    assert j.raw_score == 4 and abs(j.score - 1 / 3) < 1e-3 and j.problems == ["no holes"] and j.error is None
    assert parse_judge_reply('Sure: {"score": 10, "problems": "none"}').score == 1.0
    assert parse_judge_reply('{"score": 11}').error and parse_judge_reply('{"score": 11}').score is None
    assert parse_judge_reply("looks fine").error


def test_judge_sends_request_size_and_image():
    llm = ScriptedLLM(['{"score": 8}'])
    j = make_judge(llm=llm, model="m").judge("a disc", PNG, {"bbox_mm": [60, 60, 5]})
    assert j.score == round(7 / 9, 4)
    text, image = llm.calls[0][0]["content"]
    assert "a disc" in text["text"] and "60 x 60 x 5 mm" in text["text"] and image["png"] == PNG


def test_best_attempt_matches_pipeline_rule():
    from eval.judge import best_attempt

    v = lambda valid, wt: {"validation": {"valid": valid, "watertight": wt}}
    assert best_attempt([{"attempt": 1, "outcome": "exec_error"}, {"attempt": 2, **v(False, True)},
                         {"attempt": 3, **v(True, True)}, {"attempt": 4, **v(True, True)}]) == 3
    assert best_attempt([{"attempt": 1, **v(False, False)}, {"attempt": 2, **v(False, True)},
                         {"attempt": 3, **v(False, False)}]) == 2
    assert best_attempt([{"attempt": 1, "outcome": "exec_error"}]) is None


def test_run_judges_once_resumes_and_respects_budget(tmp_path, monkeypatch):
    import eval.judge as ej

    # Two fake benchmark parts with STLs, in a temporary results/outputs/renders tree.
    stl = tmp_path / "part.stl"
    trimesh.creation.box((10, 10, 10)).export(stl)
    rows = [dict(case=c, config="rag", rep=0, model="gen", library="lib1", category="simple", expect="build",
                 outcome="built", correct=True, bbox_mm=[10, 10, 10], stl=str(stl)) for c in ("s01", "s02")]
    (tmp_path / "gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, value in dict(RESULTS_DIR=tmp_path, JUDGE_DIR=tmp_path / "judge", RENDERS_DIR=tmp_path / "renders",
                            EVAL_DIR=tmp_path, ROOT=tmp_path).items():
        monkeypatch.setattr(ej, name, value)

    def session(replies, budget):
        judge = make_judge(llm=ScriptedLLM(replies), model="judge-m")
        return ej.Session(judge, "judge-m", "v1", 64, budget, log=lambda m: None)

    class Args:
        gen_model, config, cases, library, reps = "gen", None, None, "lib1", None

    s = session(['{"score": 9}'], budget=1)
    ej.cmd_run(Args, s)
    assert s.requests == 1 and "budget" in s.stopped
    s = session(['{"score": 2, "problems": ["wrong"]}'], budget=5)
    ej.cmd_run(Args, s)  # the first part is skipped as done; only the second is judged
    assert s.requests == 1
    out = ej.read_results(tmp_path / "judge" / "judge-m.jsonl")
    assert [(r["case"], r["score"]) for r in out] == [("s01", round(8 / 9, 4)), ("s02", round(1 / 9, 4))]
    assert (tmp_path / out[0]["render_file"]).exists()
    report = ej.summarize(ej.judged_bench(tmp_path / "judge" / "judge-m.jsonl"), threshold=0.6)
    assert "| **rag** | 2 | 50% |" in report


def test_threshold_table_and_auc():
    from eval.judge import auc, threshold_table

    data = [(0.9, True, "reference"), (0.8, True, "reference"), (0.3, False, "mismatch"), (0.6, False, "label")]
    assert auc(data) == 1.0
    best = max(threshold_table(data), key=lambda t: t["balanced"])
    assert best["threshold"] == 0.7 and best["accuracy"] == 1.0  # halfway between 0.6 (fail) and 0.8 (pass)


def test_locate_stl_finds_current_and_legacy_run_folders(tmp_path, monkeypatch):
    import eval.judge as ej

    monkeypatch.setattr(ej, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ej, "ROOT", tmp_path)
    log = {"attempt_log": [{"attempt": 1, "outcome": "exec_error"},
                           {"attempt": 2, "validation": {"valid": True, "watertight": True}}]}
    for d in (tmp_path / "bench/m/lib1/rag/s01/rep0", tmp_path / "bench/m/norag/s01/rep1"):
        (d / "attempt_2").mkdir(parents=True)
        (d / "attempt_2/model.stl").write_text("solid")
        (d / "report.json").write_text(json.dumps(log))
    row = dict(model="m", config="rag", case="s01", rep=0, library="lib1")
    assert ej.locate_stl(row) == tmp_path / "bench/m/lib1/rag/s01/rep0/attempt_2/model.stl"
    legacy = dict(model="m", config="norag", case="s01", rep=1)  # no library: old folder layout
    assert ej.locate_stl(legacy) == tmp_path / "bench/m/norag/s01/rep1/attempt_2/model.stl"
