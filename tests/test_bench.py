"""Benchmark harness with scripted LLM replies (no API calls; runs the real sandbox)."""

from eval import bench
from stl_agent.llm import ScriptedLLM
from stl_agent.retrieval import ExampleRetriever

GOOD = "PLAN:\n- block\nBBOX: 20, 10, 5\n```python\nimport cadquery as cq\nshow_object(cq.Workplane('XY').box(20, 10, 5))\n```"
WRONG_SIZE = GOOD.replace("box(20, 10, 5)", "box(30, 10, 5)").replace("BBOX: 20", "BBOX: 30")
DECLINE = "UNSUPPORTED: organic shape"

CASES = [
    {"id": "a", "category": "simple", "expect": "build", "expected_bbox_mm": [5, 10, 20], "request": "block a"},
    {"id": "b", "category": "simple", "expect": "build", "expected_bbox_mm": [20, 10, 5], "request": "block b"},
    {"id": "c", "category": "proposal", "expect": "decline", "expected_bbox_mm": None, "request": "a cat"},
]


def test_bbox_matches_any_axis_order():
    assert bench.bbox_matches([20, 10, 5], [5, 20, 10])
    assert bench.bbox_matches([20.4, 10, 5], [20, 10, 5])  # within 1 mm
    assert not bench.bbox_matches([30, 10, 5], [20, 10, 5])
    assert bench.bbox_matches(None, [1, 2, 3]) is None


def test_run_scores_resumes_and_respects_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    results = tmp_path / "results.jsonl"
    retriever = ExampleRetriever.from_file()

    # Budget 2 with 1 attempt per case: runs a and b, then stops before c.
    llm = ScriptedLLM([GOOD, WRONG_SIZE])
    s = bench.run(CASES, ["rag"], llm, "test-model", retriever, results, budget=2, max_attempts=1, log=lambda m: None)
    assert (s.ran, s.requests) == (2, 2) and s.stopped.startswith("budget")
    rows = bench.read_results(results)
    assert [(r["case"], r["outcome"], r["correct"], r["bbox_ok"]) for r in rows] == [
        ("a", "built", True, True), ("b", "built", False, False)]
    assert rows[0]["first_try"] and rows[0]["requests"] == 1

    # Second run resumes: a and b are skipped, only c runs.
    llm = ScriptedLLM([DECLINE])
    s = bench.run(CASES, ["rag"], llm, "test-model", retriever, results, budget=5, max_attempts=1, log=lambda m: None)
    assert (s.ran, s.skipped, s.stopped) == (1, 2, "")
    last = bench.read_results(results)[-1]
    assert (last["case"], last["outcome"], last["correct"]) == ("c", "declined", True)

    report = bench.summarize(bench.read_results(results))
    assert "| rag | 3 | 67% |" in report


def test_norag_prompt_has_no_empty_examples_header(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    llm = ScriptedLLM([GOOD])
    bench.run(CASES[:1], ["norag"], llm, "m", ExampleRetriever.from_file(), tmp_path / "r.jsonl", budget=1, max_attempts=1,
              log=lambda m: None)
    first = llm.calls[0][0]["content"]
    assert "solved examples" not in first and first.endswith("Request: block a")


def test_rag_features_adds_the_feature_note_only_there(tmp_path, monkeypatch):
    from stl_agent.generate import FEATURE_NOTE

    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    llm = ScriptedLLM([GOOD, GOOD])
    bench.run(CASES[:1], ["rag", "rag-features"], llm, "m", ExampleRetriever.from_file(), tmp_path / "r.jsonl",
              budget=2, max_attempts=1, log=lambda m: None)
    plain, noted = (call[0]["content"] for call in llm.calls)
    assert FEATURE_NOTE not in plain and noted.endswith(FEATURE_NOTE + "Request: block a")


def test_retry_delay_parsing():
    from stl_agent.llm import MAX_QUOTA_WAIT, _retry_delay

    assert _retry_delay("Quota exceeded ... limit: 15 ... Please retry in 47.5s.") == 47.5
    assert _retry_delay('{"retryDelay": "30s"}') == 30
    assert _retry_delay("Please retry in 1h2m3s.") is None  # daily-style wait: not a short retry
    assert _retry_delay('{"retryDelay": "40000s"}') > MAX_QUOTA_WAIT


def test_rate_limiter_spaces_requests():
    from stl_agent.llm import RateLimiter

    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    rl = RateLimiter(2, clock=lambda: now[0], sleep=sleep)
    rl.wait(); rl.wait()          # two calls at t=0 go through
    assert slept == []
    now[0] = 10
    rl.wait()                     # third call within the minute waits until t=60.5
    assert slept == [50.5] and now[0] == 60.5


def test_clients_of_one_model_share_its_rate_limit():
    """The rpm setting is per model, not per client: parallel workers each make a client (finetune.collect)."""
    from stl_agent.llm import OpenAICompatLLM

    a = OpenAICompatLLM("http://x/v1", "k", "m1", rpm=7)
    b = OpenAICompatLLM("http://x/v1/", "k", "m1", rpm=7)
    c = OpenAICompatLLM("http://x/v1", "k", "m2", rpm=7)
    assert a.pacer is b.pacer and a.pacer is not c.pacer
    assert OpenAICompatLLM("http://x/v1", "k", "m1").pacer is None


def test_repeats_run_resume_and_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    results = tmp_path / "r.jsonl"
    retriever = ExampleRetriever.from_file()
    case = CASES[:1]
    # rep 0 for both configs, then rep 1: norag fails (no code), rag succeeds.
    llm = ScriptedLLM([GOOD, GOOD, GOOD, "no code here"])
    s = bench.run(case, ["rag", "norag"], llm, "m", retriever, results, budget=10, max_attempts=1,
                  log=lambda m: None, repeats=2)
    assert s.ran == 4
    rows = bench.read_results(results)
    assert [(r["config"], r["rep"], r["correct"]) for r in rows] == [
        ("rag", 0, True), ("norag", 0, True), ("rag", 1, True), ("norag", 1, False)]
    # Everything is done: a rerun skips all four.
    s = bench.run(case, ["rag", "norag"], ScriptedLLM([]), "m", retriever, results, budget=10, max_attempts=1,
                  log=lambda m: None, repeats=2)
    assert (s.ran, s.skipped) == (0, 4)
    assert "better on a (100% vs 50%)" in bench.summarize(rows)


def test_organic_suite_runs_sdf_and_thingiverse_configs(tmp_path, monkeypatch):
    import trimesh

    from stl_agent.config import SDF_EXAMPLES_PATH
    from stl_agent.sources import thingiverse as tv
    from stl_agent.sources.thingiverse import Candidate

    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    cases = bench.load_cases(bench.ORGANIC_CASES_PATH)[:1]
    assert cases[0]["category"] == "organic" and cases[0]["search"]

    def fake_download(thing_id, dest, client=None):
        trimesh.creation.icosphere().export(dest)
        return dest, []

    monkeypatch.setattr(tv, "search", lambda term, **kw: [Candidate(id=1, name="cat", url="u", thumbnail="", author="a",
                                                               license="CC-BY")])
    monkeypatch.setattr(tv, "download_stl", fake_download)
    ball = "BBOX: 20, 20, 18.5\n```python\nfrom sdf import *\nshow_object(sphere(10))\n```"  # stood on z = 0
    llm = ScriptedLLM([ball])
    results = tmp_path / "r.jsonl"
    s = bench.run(cases, ["sdf", "thingiverse"], llm, "m", None, results, budget=4, max_attempts=1, log=lambda m: None)
    assert s.ran == 2 and s.requests == 1  # the download costs no LLM request
    sdf_row, tv_row = bench.read_results(results)
    assert sdf_row["built"] and sdf_row["library"] == f"{bench.library_id(SDF_EXAMPLES_PATH)}+{bench.sdf_prompt_id()}"
    # SDF rows without examples still depend on the SDF prompt (#47), unlike CadQuery no-RAG rows
    assert bench._lib_key("sdf-norag", bench.library_for("sdf-norag")) == f"norag+{bench.sdf_prompt_id()}"
    assert bench._lib_key("sdf-norag", "lib26-cdabac") != bench._lib_key("sdf-norag", bench.library_for("sdf-norag"))
    assert all(i.startswith("sdf_") for i in sdf_row["retrieved"])
    assert tv_row["built"] and tv_row["requests"] == 0 and bench._lib_key("thingiverse", tv_row["library"]) == "any"
    assert {c["id"] for c in bench.all_cases()} >= {"s01", "o01"}


def test_study_rows_and_parts_stay_in_their_own_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    path = bench.results_dir("s1") / "m.jsonl"
    assert path == tmp_path / "results/studies/s1/m.jsonl" and bench.results_dir() == tmp_path / "results"
    bench.run(CASES[1:2], ["rag"], ScriptedLLM([GOOD]), "m", ExampleRetriever.from_file(), path, budget=1,
              max_attempts=1, log=lambda m: None, study="s1")
    row = bench.read_results(path)[0]
    assert row["study"] == "s1" and bench.part_id(row).startswith("s1/m/")
    assert (tmp_path / "out/bench" / bench.part_id(row) / "report.json").exists()
    assert not (tmp_path / "results/m.jsonl").exists()  # earlier results untouched
    assert bench.part_id({k: v for k, v in row.items() if k != "study"}).startswith("m/")  # old ids unchanged


def test_rows_record_prompt_code_version_provider_and_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    path = tmp_path / "r.jsonl"
    bench.run(CASES[1:2], ["rag-size"], ScriptedLLM([GOOD]), "m", ExampleRetriever.from_file(), path, budget=1,
              max_attempts=1, log=lambda m: None)
    row = bench.read_results(path)[0]
    assert row["prompt"] == bench.prompt_id("rag-size") and row["prompt"].startswith("cqp-")
    assert bench.prompt_id("sdf").startswith("sdfp-") and bench.prompt_id("thingiverse") is None
    assert row["git"] and row["provider"]
    assert row["settings"]["max_attempts"] == 1 and row["settings"]["size_feedback"] == "measured"


def test_rows_split_build_time(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    llm = ScriptedLLM([GOOD.replace("box(20, 10, 5)", "box(20, 10, 5).no_such_method()"), GOOD])
    llm.request_seconds = llm.wait_seconds = 0.0

    def timed(system, messages, complete=llm.complete):
        llm.request_seconds += 2.0
        llm.wait_seconds += 0.5
        return complete(system, messages)

    llm.complete = timed
    path = tmp_path / "r.jsonl"
    bench.run(CASES[1:2], ["rag"], llm, "m", ExampleRetriever.from_file(), path, budget=2, max_attempts=2,
              log=lambda m: None)
    row = bench.read_results(path)[0]
    assert (row["llm_seconds"], row["wait_seconds"]) == (4.0, 1.0)
    assert 0 < row["sandbox_seconds"] <= row["seconds"] + 0.05  # seconds is rounded to 0.1


def test_spending_cap_counts_earlier_runs_and_lost_spend(tmp_path, monkeypatch):
    import json

    import pytest

    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    path = tmp_path / "claude-haiku-5-5.jsonl"
    retriever = ExampleRetriever.from_file()
    run = lambda llm, cap, model="claude-haiku-5-5": bench.run(
        CASES[:2], ["rag"], llm, model, retriever, path, budget=9, max_attempts=1, log=lambda m: None, max_usd=cap)
    # One build of Haiku may cost up to 1 x (12k in, 3k out) = $0.0027: a $1 cap runs both cases.
    s = run(ScriptedLLM([GOOD, GOOD]), 1.0)
    assert s.ran == 2 and all(r["usd"] == 0 for r in bench.read_results(path))
    # Spend from earlier runs and spend lost to errors both count toward the cap.
    path.write_text(json.dumps(dict(case="x", config="rag", rep=0, usd=0.995)) + "\n")
    bench.record_lost_usd(path, 0.003)
    assert bench.spent_usd(path) == pytest.approx(0.998)
    s = run(ScriptedLLM([GOOD]), 1.0)
    assert s.ran == 0 and s.stopped.startswith("spending cap")
    with pytest.raises(ValueError, match="known price"):
        run(ScriptedLLM([]), 1.0, model="no-price-model")


def test_size_configs_match_the_app_defaults():
    from stl_agent.config import settings

    for config in ("rag-size", "norag-size"):
        assert bench.CONFIGS[config]["size_feedback"] == settings().size_feedback
    assert bench.CONFIGS["rag-size"]["use_rag"] and not bench.CONFIGS["norag-size"]["use_rag"]
    assert bench._lib_key("norag-size", "lib26-x") == "any"  # no examples: shared across library versions


def test_study_manifest_runs_jobs_with_their_own_settings(tmp_path, monkeypatch):
    import pytest

    from eval import study
    from stl_agent import llm as llm_mod
    from stl_agent.config import settings

    monkeypatch.setattr(bench, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    manifest = tmp_path / "s1.toml"
    manifest.write_text('''
[study]
about = "test"

[[run]]
name = "local"
model = "m"
configs = ["rag-size"]
repeats = 2
budget = 9
max_attempts = 1
set = ["provider=ollama"]
cases = ["s01"]
''')
    seen = []

    def fake_make_llm(model=None, **kw):
        seen.append((settings().provider, model))
        return ScriptedLLM([GOOD] * 2)

    monkeypatch.setattr(llm_mod, "make_llm", fake_make_llm)
    data = study.load_study(str(manifest))
    assert data["study"]["name"] == "s1"
    job = data["run"][0]
    s = study.run_job("s1", job, log=lambda m: None)
    assert seen == [("ollama", "m")] and s.ran == 2
    assert settings().provider == "gemini"  # the job's settings don't leak
    assert study.job_status("s1", job) == dict(name="local", model="m", done=2, expected=2, usd=0.0, max_usd=None)
    assert "| local | m | 2 | 2 |" in study.status_table(data)

    manifest.write_text('[[run]]\nname = "x"\nmodel = "m"\nconfigs = ["rag-size"]\n')
    with pytest.raises(ValueError, match="provider"):  # LLM jobs must name their provider
        study.load_study(str(manifest))
    manifest.write_text('[[run]]\nname = "tv"\nmodel = "thingiverse"\nsuite = "organic"\nconfigs = ["thingiverse-check"]\n')
    assert study.download_only(study.load_study(str(manifest))["run"][0])


def test_code_version_without_git_uses_the_build_commit(monkeypatch):
    # The Docker image has no .git: rows name the commit passed in at build time.
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(bench.subprocess, "run", no_git)
    monkeypatch.setenv("STL_AGENT_COMMIT", "abc1234")
    assert bench.code_version() == "abc1234"
    monkeypatch.delenv("STL_AGENT_COMMIT")
    assert bench.code_version() == "unknown"
