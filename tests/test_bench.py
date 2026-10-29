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

    monkeypatch.setattr(tv, "search", lambda term: [Candidate(id=1, name="cat", url="u", thumbnail="", author="a",
                                                               license="CC-BY")])
    monkeypatch.setattr(tv, "download_stl", fake_download)
    ball = "BBOX: 20, 20, 20\n```python\nfrom sdf import *\nshow_object(sphere(10))\n```"
    llm = ScriptedLLM([ball])
    results = tmp_path / "r.jsonl"
    s = bench.run(cases, ["sdf", "thingiverse"], llm, "m", None, results, budget=4, max_attempts=1, log=lambda m: None)
    assert s.ran == 2 and s.requests == 1  # the download costs no LLM request
    sdf_row, tv_row = bench.read_results(results)
    assert sdf_row["built"] and sdf_row["library"] == bench.library_id(SDF_EXAMPLES_PATH)
    assert all(i.startswith("sdf_") for i in sdf_row["retrieved"])
    assert tv_row["built"] and tv_row["requests"] == 0 and bench._lib_key("thingiverse", tv_row["library"]) == "any"
    assert {c["id"] for c in bench.all_cases()} >= {"s01", "o01"}
