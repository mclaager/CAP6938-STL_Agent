"""Benchmark harness (issue #7): run the benchmark requests through `generate_cad` and report results.

    python -m eval.bench run --config rag norag --budget 18
    python -m eval.bench report

`run` appends one JSON line per (case, config) to eval/results/<model>.jsonl as soon as it finishes,
so it can be stopped and resumed: finished pairs are skipped on the next run. It stops before
a case could take the run over `--budget` LLM requests (each attempt is one request), which keeps
it inside free-tier daily quotas; run it again the next day to continue. Configs are interleaved
per case, so a partial run still gives paired RAG / no-RAG comparisons.

The harness calls `generate_cad` directly (no intake gate); gate evaluation is issue #12.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from stl_agent.config import MAX_ATTEMPTS, OUTPUT_DIR
from stl_agent.pipeline import generate_cad

EVAL_DIR = Path(__file__).parent
CASES_PATH = EVAL_DIR / "cases.json"
RESULTS_DIR = EVAL_DIR / "results"

# name -> generate_cad keyword arguments
CONFIGS = {
    "rag": dict(use_rag=True, top_k=3),
    "norag": dict(use_rag=False),
    "rag-k1": dict(use_rag=True, top_k=1),
}

BBOX_REL_TOL, BBOX_ABS_TOL = 0.03, 1.0  # per dimension: within 3% or 1 mm, whichever is larger


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["cases"]


def bbox_matches(actual, expected) -> bool | None:
    """Compare sizes in any axis order. None when there's nothing to compare."""
    if not actual or not expected:
        return None
    return all(abs(a - e) <= max(BBOX_ABS_TOL, BBOX_REL_TOL * e) for a, e in zip(sorted(actual), sorted(expected)))


def score(case: dict, res) -> dict:
    """Turn a GenResult into the fields recorded for one run."""
    bbox = res.validation.bbox_mm if res.validation else None
    bbox_ok = bbox_matches(bbox, case.get("expected_bbox_mm")) if res.success else None
    declined = res.unsupported is not None
    built = res.success
    expect = case["expect"]
    if expect == "decline":
        correct = declined
    elif expect == "either":
        correct = built or declined
    else:
        correct = built and bbox_ok is not False
    if built:
        outcome = "built"
    elif declined:
        outcome = "declined"
    elif res.stl_path:
        outcome = "best_effort"  # a mesh came out but failed validation
    else:
        outcome = "failed"
    return dict(outcome=outcome, correct=correct, built=built, declined=declined, bbox_mm=bbox, bbox_ok=bbox_ok,
                first_try=built and res.attempts == 1, attempts=res.attempts,
                failures=[h.get("outcome") for h in res.history if h.get("outcome") not in ("ok", None)],
                retrieved=res.retrieved_ids, seconds=round(res.seconds, 1), error=res.error)


class CountingLLM:
    """Counts requests and tokens; everything else goes to the wrapped backend."""

    def __init__(self, llm):
        self.llm, self.requests = llm, 0

    def complete(self, system: str, messages: list[dict]) -> str:
        self.requests += 1
        return self.llm.complete(system, messages)

    def __getattr__(self, name):  # last_truncated, input_tokens, model, ...
        return getattr(self.llm, name)


@dataclass
class RunSummary:
    ran: int = 0
    skipped: int = 0
    requests: int = 0
    stopped: str = ""


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(cases: list[dict], configs: list[str], llm, model: str, retriever, results_path: Path,
        budget: int, max_attempts: int = MAX_ATTEMPTS, log=print) -> RunSummary:
    done = {(r["case"], r["config"]) for r in read_results(results_path)}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    counter = CountingLLM(llm)
    summary = RunSummary()

    for case in cases:
        for config in configs:
            if (case["id"], config) in done:
                summary.skipped += 1
                continue
            if counter.requests + max_attempts > budget:
                summary.stopped = f"budget: {counter.requests}/{budget} requests used; run again to continue"
                summary.requests = counter.requests
                return summary

            log(f"[{case['id']} / {config}] {case['request'][:70]}")
            before = (counter.requests, getattr(llm, "input_tokens", 0), getattr(llm, "output_tokens", 0))
            out_dir = OUTPUT_DIR / "bench" / model / config / case["id"]
            try:
                res = generate_cad(case["request"], counter, retriever, log=lambda m: None, out_dir=out_dir,
                                   max_attempts=max_attempts, **CONFIGS[config])
            except Exception as e:  # quota exhausted, timeouts, network: stop without recording a result
                summary.stopped = f"error, stopping: {type(e).__name__}: {e}"
                summary.requests = counter.requests
                return summary

            row = dict(case=case["id"], config=config, model=model, category=case["category"], expect=case["expect"],
                       **score(case, res),
                       requests=counter.requests - before[0],
                       input_tokens=getattr(llm, "input_tokens", 0) - before[1],
                       output_tokens=getattr(llm, "output_tokens", 0) - before[2],
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            summary.ran += 1
            log(f"    -> {row['outcome']}, correct={row['correct']}, attempts={row['attempts']}, bbox_ok={row['bbox_ok']}")

    summary.requests = counter.requests
    return summary


def summarize(rows: list[dict]) -> str:
    """Markdown tables: per config, per category, and paired RAG vs no-RAG."""
    if not rows:
        return "No results yet."

    def stats(rs):
        n = len(rs)
        built = [r for r in rs if r["built"]]
        bbox = [r for r in built if r["bbox_ok"] is not None]
        pct = lambda k, xs: f"{100 * sum(bool(r[k]) for r in xs) / len(xs):.0f}%" if xs else "–"
        return [str(n), pct("correct", rs), pct("built", rs), pct("first_try", rs),
                f"{sum(r['attempts'] for r in rs) / n:.2f}", pct("bbox_ok", bbox),
                f"{sum(r['requests'] for r in rs)}"]

    head = "| {} | n | correct | built | first try | avg attempts | bbox ok | requests |\n|---|---|---|---|---|---|---|---|"
    out = []
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        out.append(f"## {model}\n\n" + head.format("config"))
        for config in sorted({r["config"] for r in mrows}):
            out.append(f"| {config} | " + " | ".join(stats([r for r in mrows if r["config"] == config])) + " |")
        out.append("\n" + head.format("config / category"))
        for config in sorted({r["config"] for r in mrows}):
            for cat in ["simple", "medium", "hard", "proposal"]:
                rs = [r for r in mrows if r["config"] == config and r["category"] == cat]
                if rs:
                    out.append(f"| {config} / {cat} | " + " | ".join(stats(rs)) + " |")

        by = {(r["case"], r["config"]): r for r in mrows}
        paired = sorted({c for c, k in by if k == "rag" and (c, "norag") in by})
        if paired:
            rag_only = [c for c in paired if by[(c, "rag")]["correct"] and not by[(c, "norag")]["correct"]]
            norag_only = [c for c in paired if by[(c, "norag")]["correct"] and not by[(c, "rag")]["correct"]]
            fewer = sum(by[(c, "rag")]["attempts"] < by[(c, "norag")]["attempts"] for c in paired)
            more = sum(by[(c, "rag")]["attempts"] > by[(c, "norag")]["attempts"] for c in paired)
            out.append(f"\n**RAG vs no-RAG on {len(paired)} paired cases:** correct only with RAG: "
                       f"{', '.join(rag_only) or 'none'}; correct only without RAG: {', '.join(norag_only) or 'none'}. "
                       f"RAG needed fewer attempts on {fewer}, more on {more}.")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run benchmark cases")
    r.add_argument("--config", nargs="+", default=["rag", "norag"], choices=sorted(CONFIGS))
    r.add_argument("--cases", nargs="+", help="case ids (default: all)")
    r.add_argument("--category", nargs="+", help="only these categories")
    r.add_argument("--budget", type=int, default=18, help="max LLM requests this run (default 18)")
    r.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    r.add_argument("--model", help="model id (default: CAD_MODEL / provider default)")
    rep = sub.add_parser("report", help="summarize results")
    rep.add_argument("files", nargs="*", type=Path, help="results files (default: all in eval/results)")
    args = ap.parse_args(argv)

    if args.cmd == "report":
        files = args.files or sorted(RESULTS_DIR.glob("*.jsonl"))
        print(summarize([row for f in files for row in read_results(f)]))
        return

    from stl_agent.llm import make_llm
    from stl_agent.retrieval import ExampleRetriever

    cases = load_cases()
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    llm = make_llm(model=args.model) if args.model else make_llm()
    model = getattr(llm, "model", "unknown")
    path = RESULTS_DIR / f"{model}.jsonl"
    s = run(cases, args.config, llm, model, ExampleRetriever.from_file(), path, args.budget, args.max_attempts)
    print(f"\nRan {s.ran}, skipped {s.skipped} already done, used {s.requests} requests. Results: {path}")
    if s.stopped:
        print("Stopped early: " + s.stopped)
        sys.exit(2 if s.stopped.startswith("error") else 0)


if __name__ == "__main__":
    main()
