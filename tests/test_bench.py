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
