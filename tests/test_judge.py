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
        return ej.Session(judge, "judge-m", "v2", 64, budget, log=lambda m: None)

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


def test_bands_and_label_grades():
    from eval.judge import band, label_pass

    assert [band(x) for x in (1, 3, 4, 6, 7, 8, 9, 10)] == ["different", "different", "feature_wrong", "feature_wrong",
                                                             "minor", "minor", "match", "match"]
    assert band(None) is None
    assert label_pass({"grade": "minor"}) is True and label_pass({"grade": "feature_wrong"}) is False
    assert label_pass({"pass": True}) is None and label_pass({"why": "not sure yet"}) is None  # grades only


def test_band_agreement_counts_lenient_and_harsh():
    from eval.judge import agreement_table, band_agreement

    pairs = [("match", "match"), ("match", "feature_wrong"),  # the rook: judge says match, human says feature wrong
             ("minor", "match"), ("different", "different")]
    a = band_agreement(pairs)
    assert (a["n"], a["exact"], a["within_one"], a["lenient"], a["harsh"]) == (4, 0.5, 0.75, 1, 1)
    assert a["matrix"][("match", "feature_wrong")] == 1
    assert "Exact band 50%" in agreement_table(a)


def test_sheet_has_grade_buttons_and_keeps_labels_in_browser(tmp_path, monkeypatch):
    import eval.judge as ej

    row = dict(part="gen/lib1/rag/s01/rep0", judge="llm:j", prompt="v2", render="v2-256px", kind="bench", case="s01",
               config="rag", rep=0, gen_model="gen", library="lib1", category="simple", bench_outcome="built",
               bench_correct=True, render_file="renders/v2-256px/gen/lib1/rag/s01-rep0.png", score=0.8889,
               raw_score=9.0, matches=[], problems=[], error=None)
    (tmp_path / "judge").mkdir()
    (tmp_path / "judge" / "j.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(ej, "JUDGE_DIR", tmp_path / "judge")
    monkeypatch.setattr(ej, "LABELS_PATH", tmp_path / "labels.json")
    page = ej.write_sheet(tmp_path / "index.html", threshold=0.5).read_text()
    for g in ej.GRADES:
        assert f"data-g={g}" in page
    assert "localStorage" in page and "(match)" in page  # judge band shown next to the score
    report = ej.summarize([row], threshold=0.5)
    assert "| rag | 1 | 0 | 0 | 0 |" in report  # band counts: one 'match'


def test_prompts_are_files_paired_with_render_versions():
    from stl_agent.judge import DEFAULT_PROMPT
    from stl_agent.judge.llm_judge import PROMPTS, RENDER_FOR_PROMPT
    from stl_agent.render import RENDER_VERSION

    assert set(PROMPTS) == set(RENDER_FOR_PROMPT)  # every prompt file says which render it describes
    assert RENDER_FOR_PROMPT[DEFAULT_PROMPT] == RENDER_VERSION  # the default matches the current renders
    assert all("Score 1-10" in p and '"score"' in p for p in PROMPTS.values())


def test_study_parts_are_judged_into_the_study_folder(tmp_path, monkeypatch):
    import eval.judge as ej

    for name, value in dict(RESULTS_DIR=tmp_path, JUDGE_DIR=tmp_path / "judge").items():
        monkeypatch.setattr(ej, name, value)
    assert ej.split_part_id("s1/gen/lib1/rag/s01/rep2") == ("s1", "gen", "lib1", "rag", "s01", "rep2")
    assert ej.split_part_id("gen/lib1/rag/s01/rep0") == (None, "gen", "lib1", "rag", "s01", "rep0")
    judge = make_judge(llm=ScriptedLLM([]), model="judge-m")
    assert ej.Session(judge, "judge-m", "v2", 64, 1, study="s1").path == tmp_path / "studies/s1/judge/judge-m.jsonl"
    assert ej.Session(judge, "judge-m", "v2", 64, 1).path == tmp_path / "judge/judge-m.jsonl"
    row = dict(study="s1", case="s01", config="rag", rep=0, model="gen", library="lib1", outcome="built")
    (tmp_path / "studies/s1").mkdir(parents=True)
    (tmp_path / "studies/s1/gen.jsonl").write_text(json.dumps(row) + "\n")
    assert ej.bench_rows("gen", None, None, "lib1", study="s1") == [row]
    assert ej.bench_rows("gen", None, None, "lib1") == []


def test_judge_spending_cap(tmp_path, monkeypatch):
    import pytest

    import eval.judge as ej

    monkeypatch.setattr(ej, "JUDGE_DIR", tmp_path / "judge")
    judge = make_judge(llm=ScriptedLLM(['{"score": 9}'] * 3), model="claude-haiku-5-5")
    s = ej.Session(judge, "claude-haiku-5-5", "v2", 64, budget=5, log=lambda m: None, max_usd=0.004)
    assert s.judge_one("p1", "a block", b"png", {}) is not None  # reserve 4k+4k tokens = $0.0024
    assert s.judge_one("p2", "a block", b"png", {}) is not None
    ej.record_lost_usd(s.path, 0.002)
    s = ej.Session(judge, "claude-haiku-5-5", "v2", 64, budget=5, log=lambda m: None, max_usd=0.004)
    assert s.judge_one("p3", "a block", b"png", {}) is None and s.stopped.startswith("spending cap")
    with pytest.raises(ValueError, match="known price"):
        ej.Session(judge, "no-price", "v2", 64, budget=5, max_usd=1.0)


def _bench_tree(tmp_path, monkeypatch, models=("gen",), study=None):
    """Fake benchmark parts with STLs in a temporary results/outputs/renders tree."""
    import eval.judge as ej

    stl = tmp_path / "part.stl"
    trimesh.creation.box((10, 10, 10)).export(stl)
    folder = tmp_path / "studies" / study if study else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    for model in models:
        rows = [dict(case=c, config="rag", rep=0, model=model, library="lib1", category="simple", expect="build",
                     outcome="built", correct=True, bbox_mm=[10, 10, 10], stl=str(stl), **({"study": study} if study else {}))
                for c in ("s01", "s02", "s03")]
        (folder / f"{model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, value in dict(RESULTS_DIR=tmp_path, JUDGE_DIR=tmp_path / "judge", RENDERS_DIR=tmp_path / "renders",
                            EVAL_DIR=tmp_path, ROOT=tmp_path, LABELS_PATH=tmp_path / "labels.json").items():
        monkeypatch.setattr(ej, name, value)
    return ej


def test_judge_repeats_are_recorded_and_resumed(tmp_path, monkeypatch):
    ej = _bench_tree(tmp_path, monkeypatch)

    class Args:
        gen_model, config, cases, library, reps, repeats = "gen", None, None, "lib1", None, 2

    s = ej.Session(make_judge(llm=ScriptedLLM(['{"score": 9}'] * 6), model="j"), "j", "v2", 64, 9, log=lambda m: None)
    ej.cmd_run(Args, s)
    rows = ej.read_results(s.path)
    assert sorted((r["case"], r["judge_rep"]) for r in rows) == [(c, k) for c in ("s01", "s02", "s03") for k in (0, 1)]
    s = ej.Session(make_judge(llm=ScriptedLLM([]), model="j"), "j", "v2", 64, 9, log=lambda m: None)
    ej.cmd_run(Args, s)  # everything done: no requests
    assert s.requests == 0 and not s.stopped


def test_pick_labels_spreads_over_builders_and_verdicts(tmp_path, monkeypatch):
    ej = _bench_tree(tmp_path, monkeypatch, models=("gemini-x", "local-y"), study="s1")
    judge_rows = []
    for model in ("gemini-x", "local-y"):
        for c, score in (("s01", 0.9), ("s02", 0.9), ("s03", 0.1)):
            judge_rows.append(dict(kind="bench", part=f"s1/{model}/lib1/rag/{c}/rep0", score=score))
    (tmp_path / "studies/s1/judge").mkdir(parents=True)
    (tmp_path / "studies/s1/judge/j.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judge_rows))
    picked = ej.pick_labels("s1", 4, 0, "j", 0.5)
    assert picked == ej.pick_labels("s1", 4, 0, "j", 0.5)  # deterministic
    assert {p.split("/")[1] for p in picked} == {"gemini-x", "local-y"}
    assert sum(p.endswith("s03/rep0") for p in picked) == 2  # both failed parts (one per builder)
    assert ej.add_picked_labels(picked, "test") == 4 and ej.add_picked_labels(picked, "test") == 0
    assert not set(ej.pick_labels("s1", 9, 0, "j", 0.5)) & set(picked)  # already picked parts are skipped


def test_blind_sheet_shows_only_parts_to_grade_without_scores(tmp_path, monkeypatch):
    ej = _bench_tree(tmp_path, monkeypatch)
    row = dict(part="gen/lib1/rag/s01/rep0", judge="llm:j", prompt="v2", render="v2-256px", kind="bench", case="s01",
               config="rag", rep=0, gen_model="gen", library="lib1", category="simple", bench_outcome="built",
               bench_correct=True, render_file="renders/v2-256px/gen/lib1/rag/s01-rep0.png", score=0.8889,
               raw_score=9.0, matches=[], problems=["a telltale problem"], error=None)
    other = dict(row, part="gen/lib1/rag/s02/rep0", case="s02", render_file="renders/v2-256px/gen/lib1/rag/s02-rep0.png")
    (tmp_path / "judge").mkdir()
    (tmp_path / "judge/j.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    ej.add_picked_labels(["gen/lib1/rag/s01/rep0"], "test")
    page = ej.write_sheet(tmp_path / "blind.html", 0.5, blind=True).read_text()
    assert "s01-rep0.png" in page and "s02-rep0.png" not in page
    assert "telltale" not in page and "0.89" not in page and "data-g=match" in page
    assert ">#1<" in page and "measured size" in page
    assert "<h2>Parts to grade (1)</h2>" in page and "<h2>gen" not in page  # no builder / config headings
    assert "<div class=id>#1</div>" in page and "<div class=id>gen/" not in page
    assert "telltale" in ej.write_sheet(tmp_path / "all.html", 0.5).read_text()


def test_compare_judges_tables():
    from eval.judge import compare_judges

    passing = {1, 2, 5, 6}  # both halves of the cases (split alternately) hold passes and fails
    labels = {f"gemini-x/l/rag/c{i}/rep0": {"grade": "match" if i in passing else "different"} for i in range(8)}
    rows = {"good": [], "bad": []}
    for i in range(8):
        part, truth = f"gemini-x/l/rag/c{i}/rep0", i in passing
        for k in (0, 1):
            common = dict(part=part, kind="bench", case=f"c{i}", gen_model="gemini-x", judge_rep=k, seconds=2.0,
                          usd=0.001, input_tokens=1500, output_tokens=80)
            rows["good"].append(dict(common, score=0.9 if truth else 0.1, raw_score=9.0 if truth else 1.0))
            rows["bad"].append(dict(common, score=0.9, raw_score=9.0))  # passes everything
    report = compare_judges(rows, labels, baseline="bad")
    assert "| good | 8 | 2 | 1.00 | 100% [100-100] | 0.50, 0.50 |" in report
    assert "| 1,500 | 80 | $0.0010 | 2.0 |" in report
    assert "| good | - | 100% [100-100] (8) | - |" in report  # all graded mechanical parts, held-out
    assert "| bad | 8 | 2 | 0.50 | 50% [" in report
    assert "| good | +0 pts (8) | - | - |" in report and "| bad | +50 pts (8) |" in report
    assert "| bad | good | 8 | 0 | 4 |" in report
    assert "### Items bad gets wrong (4)" in report


def test_render_parts_ahead_of_judging(tmp_path, monkeypatch):
    ej = _bench_tree(tmp_path, monkeypatch, study="s1")
    assert ej.render_parts("s1", px=64, log=lambda m: None) == {"rendered": 3}
    assert ej.render_parts("s1", px=64, log=lambda m: None) == {"done": 3}  # saved renders are reused
    assert (tmp_path / "renders" / ej.render_tag(64) / "s1/gen/lib1/rag/s01-rep0.png").exists()


def test_every_judge_command_parses(tmp_path, monkeypatch):
    """Each subcommand gets the options main() reads for all of them (--set, --study)."""
    ej = _bench_tree(tmp_path, monkeypatch, study="s1")
    ej.main(["render", "--study", "s1", "--px", "64"])
    assert (tmp_path / "renders" / ej.render_tag(64) / "s1/gen/lib1/rag/s01-rep0.png").exists()
    ej.main(["report", "--study", "s1"])
    ej.main(["compare", "--study", "s1", "--judge-model", "j"])
