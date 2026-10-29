"""Benchmark harness (issue #7): run the benchmark requests through `generate_cad` and report results.

    python -m eval.bench run --config rag norag --budget 18
    python -m eval.bench report

`run` appends one JSON line per (case, config) to eval/results/<model>.jsonl as soon as it finishes,
so it can be stopped and resumed: finished pairs are skipped on the next run. It stops before
a case could take the run over `--budget` LLM requests (each attempt is one request), which keeps
it inside free-tier daily quotas; run it again the next day to continue. Configs are interleaved
per case, so a partial run still gives paired RAG / no-RAG comparisons.

The harness calls `generate_cad` directly (no intake gate); gate evaluation is issue #12.

    python -m eval.bench run --suite organic [--config sdf sdf-norag thingiverse]

runs the organic cases (eval/organic_cases.json, #42): "sdf" sculpts them with SDF code, and "thingiverse"
records the first Thingiverse model that downloads for each case's search words (no LLM requests), for the
visual judge to compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from stl_agent.config import (EXAMPLES_PATH, OUTPUT_DIR, ROOT, SDF_EXAMPLES_PATH, add_set_argument, override_settings,
                               settings)
from stl_agent.generate import generate_cad

EVAL_DIR = Path(__file__).parent
CASES_PATH = EVAL_DIR / "cases.json"
ORGANIC_CASES_PATH = EVAL_DIR / "organic_cases.json"  # #42
RESULTS_DIR = EVAL_DIR / "results"

# name -> generate_cad keyword arguments
CONFIGS = {
    # Every config is pinned to the settings all earlier rows were recorded with, whatever the app default is:
    # no API hints and the general size-failure message (size_feedback "fix"). "rag-hints" adds API hints;
    # "rag-size" uses the size-failure message the app has used since #23.
    "rag": dict(use_rag=True, top_k=3, api_hints=False, size_feedback="fix"),
    "rag-hints": dict(use_rag=True, top_k=3, api_hints=True, size_feedback="fix"),
    "rag-size": dict(use_rag=True, top_k=3, api_hints=False, size_feedback="measured"),
    "norag": dict(use_rag=False, api_hints=False, size_feedback="fix"),
    "rag-k1": dict(use_rag=True, top_k=1, api_hints=False, size_feedback="fix"),  # top-k ablation (#7): 1 and 5 examples vs the default 3
    "rag-k5": dict(use_rag=True, top_k=5, api_hints=False, size_feedback="fix"),
    # "retriever" is a stl_agent.retrieval kind, default "tfidf". Older results also contain
    # rag-t20 / rag-t30 (similarity cutoffs, removed: they never beat plain rag) and rag-ext (built-in library
    # plus a Text-to-CadQuery pool, removed: no gain, see eval/results/external-rag.md).
    "rag-char": dict(use_rag=True, top_k=3, retriever="tfidf-char", api_hints=False, size_feedback="fix"),
    # Organic cases (#42): SDF scripts with the SDF example library ("sdf" retriever), or without examples.
    "sdf": dict(language="sdf", use_rag=True, top_k=3, retriever="sdf", size_feedback="measured"),
    "sdf-norag": dict(language="sdf", use_rag=False, size_feedback="measured"),
    "thingiverse": dict(download=True),  # no generation: the first Thingiverse model that downloads
}
SUITES = {"main": (CASES_PATH, ["rag", "norag"]), "organic": (ORGANIC_CASES_PATH, ["sdf", "sdf-norag"])}


def library_id(path: Path = EXAMPLES_PATH) -> str:
    """Names the example library version, e.g. "lib26-3f2a1c", so results from different
    libraries are kept apart. Rows recorded before this existed are "lib9"."""
    data = Path(path).read_bytes()
    return f"lib{len(json.loads(data))}-{hashlib.sha1(data).hexdigest()[:6]}"


def library_for(config: str) -> str:
    """The example library a config retrieves from: the SDF one for SDF configs."""
    return library_id(SDF_EXAMPLES_PATH) if CONFIGS.get(config, {}).get("retriever") == "sdf" else library_id()


def _lib_key(config: str, library: str) -> str:
    """no-RAG and download configs ignore the library."""
    cfg = CONFIGS.get(config, {})
    return "any" if not cfg.get("use_rag", True) or cfg.get("download") else library

BBOX_REL_TOL, BBOX_ABS_TOL = 0.03, 1.0  # per dimension: within 3% or 1 mm, whichever is larger


def _repo_relative(path) -> str | None:
    """The part's STL path relative to the repo root (outputs/ is gitignored, so local only)."""
    if not path:
        return None
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["cases"]


def all_cases() -> list[dict]:
    """The main and the organic cases, for looking a result's case up by id."""
    return load_cases() + load_cases(ORGANIC_CASES_PATH)


def download_first(case: dict, out_dir: Path):
    """The "thingiverse" config: the first result for the case's search words that downloads, as a GenResult
    (no LLM requests). Downloaded meshes are often not watertight; that shows as a best-effort part."""
    from stl_agent.generate import GenResult
    from stl_agent.sandbox import validate_mesh
    from stl_agent.sources import thingiverse as tv

    t0, res = time.time(), GenResult(success=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in tv.search(case["search"]):
        try:
            path, _ = tv.download_stl(c.id, out_dir / "model.stl")
        except tv.ThingiverseError:
            continue
        res.validation = validate_mesh(str(path))
        res.success, res.stl_path, res.code = res.validation.valid, str(path), f"# {c.credit()}\n"
        res.error = None if res.success else "Downloaded mesh: " + "; ".join(res.validation.issues)
        (out_dir / "source.json").write_text(json.dumps(c.to_dict(), indent=1))
        break
    else:
        res.error = f"No Thingiverse result downloaded for {case['search']!r}."
    res.seconds = time.time() - t0
    return res


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
        budget: int, max_attempts: int | None = None, log=print, repeats: int = 1,
        library: str | None = None) -> RunSummary:
    """Run every (repeat, case, config) not yet in the results file. Repeats go in the outer loop, so a
    partial run covers all cases evenly; rows from before repeats existed count as repeat 0."""
    max_attempts = max_attempts or settings().max_attempts
    done = {(r["case"], r["config"], r.get("rep", 0), _lib_key(r["config"], r.get("library", "lib9")))
            for r in read_results(results_path)}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    counter = CountingLLM(llm)
    summary = RunSummary()
    retrievers = {"tfidf": retriever}  # other kinds are built on first use

    for case, config, rep in ((c, k, n) for n in range(repeats) for c in cases for k in configs):
        lib = library or library_for(config)
        if (case["id"], config, rep, _lib_key(config, lib)) in done:
            summary.skipped += 1
            continue
        if counter.requests + max_attempts > budget:
            summary.stopped = f"budget: {counter.requests}/{budget} requests used; run again to continue"
            summary.requests = counter.requests
            return summary

        log(f"[{case['id']} / {config} / rep {rep}] {case['request'][:60]}")
        before = (counter.requests, llm.input_tokens, llm.output_tokens)
        out_dir = OUTPUT_DIR / "bench" / model / lib / config / case["id"] / f"rep{rep}"
        try:
            kw = dict(CONFIGS[config])
            kind = kw.pop("retriever", "tfidf")
            if kw.pop("download", False):
                res = download_first(case, out_dir)
            else:
                if kind not in retrievers:
                    from stl_agent.retrieval import make_retriever

                    retrievers[kind] = (make_retriever(path=SDF_EXAMPLES_PATH) if kind == "sdf"
                                        else make_retriever(kind))
                res = generate_cad(case["request"], counter, retrievers[kind], log=lambda m: None, out_dir=out_dir,
                                   max_attempts=max_attempts, **kw)
        except Exception as e:  # quota exhausted, timeouts, network: stop without recording a result
            summary.stopped = f"error, stopping: {type(e).__name__}: {e}"
            summary.requests = counter.requests
            return summary

        row = dict(case=case["id"], config=config, rep=rep, model=model, library=lib,
                   category=case["category"], expect=case["expect"],
                   **score(case, res),
                   requests=counter.requests - before[0],
                   input_tokens=llm.input_tokens - before[1],
                   output_tokens=llm.output_tokens - before[2],
                   stl=_repo_relative(res.stl_path),
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
    groups = []
    for model in sorted({r["model"] for r in rows}):
        libs = sorted({r.get("library", "lib9") for r in rows if r["model"] == model and _lib_key(r["config"], "x") != "any"})
        for lib in libs or ["any"]:
            groups.append((model, lib, [r for r in rows if r["model"] == model and
                                        (_lib_key(r["config"], "x") == "any" or r.get("library", "lib9") == lib)]))
    for model, lib, mrows in groups:
        out.append(f"## {model} — {lib} (no-RAG rows are shared across libraries)\n\n" + head.format("config"))
        for config in sorted({r["config"] for r in mrows}):
            out.append(f"| {config} | " + " | ".join(stats([r for r in mrows if r["config"] == config])) + " |")
        out.append("\n" + head.format("config / category"))
        for config in sorted({r["config"] for r in mrows}):
            known = ["simple", "medium", "hard", "proposal", "casual"]
            for cat in known + sorted({r["category"] for r in mrows} - set(known)):
                rs = [r for r in mrows if r["config"] == config and r["category"] == cat]
                if rs:
                    out.append(f"| {config} / {cat} | " + " | ".join(stats(rs)) + " |")

        # Per-case success rate (over repeats) of each RAG config against no-RAG.
        rate = {}
        for r in mrows:
            rate.setdefault((r["case"], r["config"]), []).append(bool(r["correct"]))
        for config in sorted({k for _, k in rate if k != "norag"}):
            paired = sorted({c for c, k in rate if k == config and (c, "norag") in rate})
            if not paired:
                continue
            frac = lambda c, k: sum(rate[(c, k)]) / len(rate[(c, k)])
            better = [f"{c} ({frac(c, config):.0%} vs {frac(c, 'norag'):.0%})" for c in paired if frac(c, config) > frac(c, "norag")]
            worse = [f"{c} ({frac(c, config):.0%} vs {frac(c, 'norag'):.0%})" for c in paired if frac(c, config) < frac(c, "norag")]
            out.append(f"\n**{config} vs norag, per-case success rate over {len(paired)} cases:** "
                       f"better on {', '.join(better) or 'none'}; worse on {', '.join(worse) or 'none'}.")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run benchmark cases")
    r.add_argument("--suite", choices=sorted(SUITES), default="main",
                   help="main: eval/cases.json; organic: eval/organic_cases.json (#42)")
    r.add_argument("--config", nargs="+", choices=sorted(CONFIGS), help="default: rag norag, or sdf sdf-norag for organic")
    r.add_argument("--cases", nargs="+", help="case ids (default: all)")
    r.add_argument("--category", nargs="+", help="only these categories")
    r.add_argument("--budget", type=int, default=18, help="max LLM requests this run (default 18)")
    r.add_argument("--max-attempts", type=int, help="default: the max_attempts setting")
    r.add_argument("--model", help="model id (default: the provider's model in settings)")
    r.add_argument("--repeats", type=int, default=1, help="runs per (case, config), to average out randomness")
    add_set_argument(r)
    rep = sub.add_parser("report", help="summarize results")
    rep.add_argument("files", nargs="*", type=Path, help="results files (default: all in eval/results)")
    args = ap.parse_args(argv)

    if args.cmd == "report":
        files = args.files or sorted(RESULTS_DIR.glob("*.jsonl"))
        print(summarize([row for f in files for row in read_results(f)]))
        return

    override_settings(args.set)
    from stl_agent.llm import make_llm
    from stl_agent.retrieval import ExampleRetriever

    cases_path, default_configs = SUITES[args.suite]
    cases = load_cases(cases_path)
    configs = args.config or default_configs
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    llm = make_llm(model=args.model) if args.model else make_llm()
    model = getattr(llm, "model", "unknown")
    path = RESULTS_DIR / f"{model}.jsonl"
    s = run(cases, configs, llm, model, ExampleRetriever.from_file(), path, args.budget, args.max_attempts,
            repeats=args.repeats)
    print(f"\nRan {s.ran}, skipped {s.skipped} already done, used {s.requests} requests. Results: {path}")
    if s.stopped:
        print("Stopped early: " + s.stopped)
        sys.exit(2 if s.stopped.startswith("error") else 0)


if __name__ == "__main__":
    main()
