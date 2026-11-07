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
import subprocess
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


def results_dir(study: str | None = None) -> Path:
    """Where result rows go: eval/results, or eval/results/studies/<study> for a self-contained round (e.g. the
    report round), so its rows never mix with earlier ones."""
    return RESULTS_DIR / "studies" / study if study else RESULTS_DIR


def part_id(row: dict) -> str:
    """Stable id of a benchmark part, e.g. "gemini-3.5-flash-lite/lib26-cdabac/rag/m04/rep0". Parts of a study
    start with its name ("report-v1/gemini-3.5-flash-lite/..."). Also its folder under outputs/bench/."""
    pid = f"{row['model']}/{row.get('library') or 'lib9'}/{row['config']}/{row['case']}/rep{row.get('rep', 0)}"
    return f"{row['study']}/{pid}" if row.get("study") else pid

# name -> generate_cad keyword arguments
CONFIGS = {
    # Every config is pinned to the settings all earlier rows were recorded with, whatever the app default is:
    # no API hints and the general size-failure message (size_feedback "fix"). "rag-hints" adds API hints;
    # "rag-size" uses the size-failure message the app has used since #23.
    "rag": dict(use_rag=True, top_k=3, api_hints=False, size_feedback="fix"),
    "rag-hints": dict(use_rag=True, top_k=3, api_hints=True, size_feedback="fix"),
    "rag-size": dict(use_rag=True, top_k=3, api_hints=False, size_feedback="measured"),
    "norag": dict(use_rag=False, api_hints=False, size_feedback="fix"),
    # The app's current setup, with and without examples (the report round compares these two).
    "norag-size": dict(use_rag=False, api_hints=False, size_feedback="measured"),
    "rag-k1": dict(use_rag=True, top_k=1, api_hints=False, size_feedback="fix"),  # top-k ablation (#7): 1 and 5 examples vs the default 3
    "rag-k5": dict(use_rag=True, top_k=5, api_hints=False, size_feedback="fix"),
    # "retriever" is a stl_agent.retrieval kind, default "tfidf". Older results also contain
    # rag-t20 / rag-t30 (similarity cutoffs, removed: they never beat plain rag) and rag-ext (built-in library
    # plus a Text-to-CadQuery pool, removed: no gain, see eval/results/external-rag.md).
    "rag-char": dict(use_rag=True, top_k=3, retriever="tfidf-char", api_hints=False, size_feedback="fix"),
    # #29: rag plus a line asking for every requested feature even if the examples lack it (generate.FEATURE_NOTE).
    "rag-features": dict(use_rag=True, top_k=3, api_hints=False, size_feedback="fix", feature_note=True),
    # Organic cases (#42): SDF scripts with the SDF example library ("sdf" retriever), or without examples.
    "sdf": dict(language="sdf", use_rag=True, top_k=3, retriever="sdf", size_feedback="measured"),
    "sdf-norag": dict(language="sdf", use_rag=False, size_feedback="measured"),
    "thingiverse": dict(download=True),  # no generation: the first Thingiverse model that downloads
    # ...the first named or tagged as the case's subject (#39), as the app's automatic pick does
    "thingiverse-check": dict(download=True, check=True),
}
SUITES = {"main": (CASES_PATH, ["rag", "norag"]), "organic": (ORGANIC_CASES_PATH, ["sdf", "sdf-norag"])}


def library_id(path: Path = EXAMPLES_PATH) -> str:
    """Names the example library version, e.g. "lib26-3f2a1c", so results from different
    libraries are kept apart. Rows recorded before this existed are "lib9"."""
    data = Path(path).read_bytes()
    return f"lib{len(json.loads(data))}-{hashlib.sha1(data).hexdigest()[:6]}"


def sdf_prompt_id() -> str:
    """Names the SDF system prompt version, e.g. "sdfp-1a2b3c" (#47 changed it). Rows from before this existed
    have none."""
    from stl_agent.generate import SYSTEM_PROMPT_SDF

    return "sdfp-" + hashlib.sha1(SYSTEM_PROMPT_SDF.encode()).hexdigest()[:6]


def prompt_id(config: str) -> str | None:
    """Names the build system prompt a config sends, e.g. "cqp-1a2b3c" (CadQuery) or "sdfp-1a2b3c" (SDF), so a
    prompt change shows in the rows. None for download configs (no prompt)."""
    from stl_agent.generate import SYSTEM_PROMPT

    cfg = CONFIGS.get(config, {})
    if cfg.get("download"):
        return None
    if cfg.get("language") == "sdf":
        return sdf_prompt_id()
    return "cqp-" + hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:6]


# Files the eval commands write while they run; changes there don't make the code "dirty".
DATA_PATHS = ("eval/results", "eval/renders", "eval/judge_labels.json")


def code_version() -> str:
    """Short git commit of the checkout, plus "+dirty" when tracked code has uncommitted changes (result and
    render files excepted), so every row names the code that produced it."""
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", *(f":!{p}" for p in DATA_PATHS))
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


def library_for(config: str) -> str:
    """The example library a config retrieves from: the SDF one for SDF configs, plus the SDF prompt version
    ("lib11-5cd03c+sdfp-1a2b3c", or "norag+sdfp-1a2b3c" without examples)."""
    cfg = CONFIGS.get(config, {})
    if cfg.get("language") == "sdf":
        lib = library_id(SDF_EXAMPLES_PATH) if cfg.get("use_rag", True) else "norag"
        return f"{lib}+{sdf_prompt_id()}"
    return library_id()


def _lib_key(config: str, library: str) -> str:
    """no-RAG and download configs ignore the library, except SDF ones: their key holds the prompt version."""
    cfg = CONFIGS.get(config, {})
    if cfg.get("language") == "sdf":
        return library
    return "any" if not cfg.get("use_rag", True) or cfg.get("download") else library

# Spending caps (--max-usd) assume a build still to run costs up to max_attempts requests of 12k input and 3k
# output tokens (also used by finetune.collect).
RESERVE_TOKENS = (12_000, 3_000)


def lost_usd_path(results_path: Path) -> Path:
    """Spend that has no result row (a build stopped by an error after some requests): <model>.lost-usd.json
    beside the results, so a spending cap still counts it after a restart."""
    return results_path.with_name(results_path.stem + ".lost-usd.json")


def spent_usd(results_path: Path) -> float:
    """Dollars recorded so far for this results file: its rows' usd plus spend lost to errors."""
    lost = lost_usd_path(results_path)
    return (sum(r.get("usd") or 0 for r in read_results(results_path))
            + (sum(json.loads(lost.read_text())) if lost.exists() else 0.0))


def record_lost_usd(results_path: Path, usd: float | None) -> None:
    if usd:
        lost = lost_usd_path(results_path)
        lost.parent.mkdir(parents=True, exist_ok=True)
        lost.write_text(json.dumps((json.loads(lost.read_text()) if lost.exists() else []) + [round(usd, 4)]))


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


def download_first(case: dict, out_dir: Path, check: bool = False):
    """The "thingiverse" config: the first result for the case's search words that downloads, as a GenResult
    (no LLM requests). Downloaded meshes are often not watertight; that shows as a best-effort part.
    `check` ("thingiverse-check", #39): only results named or tagged as the case's subject."""
    from stl_agent.generate import GenResult
    from stl_agent.sandbox import validate_mesh
    from stl_agent.sources import thingiverse as tv

    t0, res = time.time(), GenResult(success=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    words = tv.object_words(case.get("subject", ""), case["search"]) if check else None
    for c in tv.search(case["search"], words=words, strict=check):
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
        res.error = (f"No Thingiverse result downloaded for {case['search']!r}"
                     + (f" named or tagged {'/'.join(sorted(words))}." if check else "."))
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


def _timers(llm) -> tuple[float, float]:
    """The backend's running (request, wait) seconds; 0 for backends that don't measure them."""
    return getattr(llm, "request_seconds", 0.0), getattr(llm, "wait_seconds", 0.0)


def _timing(res, llm, before: tuple[float, float]) -> dict:
    """Where one build's time went: `llm_seconds` inside requests, `wait_seconds` rate pacing and retry back-off,
    `sandbox_seconds` running and checking scripts. Output tokens per second follow from the row's tokens."""
    llm_s, wait_s = (now - then for now, then in zip(_timers(llm), before))
    return dict(llm_seconds=round(llm_s, 2), wait_seconds=round(wait_s, 2),
                sandbox_seconds=round(getattr(res, "sandbox_seconds", 0.0), 2))


def study_tables(study_name: str, rows: list[dict], judge_model: str | None, threshold: float | None,
                 organic_judge_model: str | None = None) -> str:
    """The study report (eval/study.py): its manifest's comparisons, and the judge's verdicts when it has judged
    the study's parts."""
    from eval import study
    from stl_agent.judge import judge_model as default_judge_model

    name = judge_model or default_judge_model()
    judge_rows = read_results(results_dir(study_name) / "judge" / f"{name}.jsonl")
    verdicts = study.judge_verdicts(judge_rows, settings().judge_threshold if threshold is None else threshold)
    manifest = study.STUDIES_DIR / f"{study_name}.toml"
    compare = study.load_study(str(manifest)).get("compare", []) if manifest.exists() else []
    organic = None
    if organic_judge_model:
        organic = study.judge_verdicts(read_results(results_dir(study_name) / "judge" / f"{organic_judge_model}.jsonl"),
                                       settings().judge_threshold if threshold is None else threshold)
    return study.study_report(rows, verdicts, compare, judge_name=name, organic_verdicts=organic,
                              organic_judge_name=organic_judge_model or "")


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
        library: str | None = None, study: str | None = None, max_usd: float | None = None) -> RunSummary:
    """Run every (repeat, case, config) not yet in the results file. Repeats go in the outer loop, so a
    partial run covers all cases evenly; rows from before repeats existed count as repeat 0. `study` is recorded
    on each row and keeps its parts in their own folder (outputs/bench/<study>/...). `max_usd`: stop before a build
    could take this results file's total spend (earlier runs included) over that many dollars."""
    from stl_agent.llm import price_usd

    max_attempts = max_attempts or settings().max_attempts
    done = {(r["case"], r["config"], r.get("rep", 0), _lib_key(r["config"], r.get("library", "lib9")))
            for r in read_results(results_path)}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    counter = CountingLLM(llm)
    summary = RunSummary()
    # Recorded on every row: the code, provider and settings that produced it.
    meta = dict(git=code_version(), provider=settings().provider)
    reserve = price_usd(model, *RESERVE_TOKENS)
    if max_usd is not None and reserve is None:
        raise ValueError(f"--max-usd needs a known price; {model} is not in stl_agent.llm.PRICES")
    spent = spent_usd(results_path)
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
        if max_usd is not None and spent + max_attempts * reserve > max_usd:
            summary.stopped = (f"spending cap: ${spent:.2f} of ${max_usd:.2f} spent; the next build could cost up to "
                               f"${max_attempts * reserve:.2f}")
            summary.requests = counter.requests
            return summary

        log(f"[{case['id']} / {config} / rep {rep}] {case['request'][:60]}")
        before = (counter.requests, llm.input_tokens, llm.output_tokens, *_timers(llm))
        tag = {"study": study} if study else {}
        out_dir = OUTPUT_DIR / "bench" / part_id(dict(tag, model=model, library=lib, config=config, case=case["id"], rep=rep))
        try:
            kw = dict(CONFIGS[config])
            kind = kw.pop("retriever", "tfidf")
            if kw.pop("download", False):
                res = download_first(case, out_dir, check=kw.pop("check", False))
            else:
                if kind not in retrievers:
                    from stl_agent.retrieval import make_retriever

                    retrievers[kind] = (make_retriever(path=SDF_EXAMPLES_PATH) if kind == "sdf"
                                        else make_retriever(kind))
                res = generate_cad(case["request"], counter, retrievers[kind], log=lambda m: None, out_dir=out_dir,
                                   max_attempts=max_attempts, **kw)
        except Exception as e:  # quota exhausted, timeouts, network: stop without recording a result
            record_lost_usd(results_path, price_usd(model, llm.input_tokens - before[1], llm.output_tokens - before[2]))
            summary.stopped = f"error, stopping: {type(e).__name__}: {e}"
            summary.requests = counter.requests
            return summary

        run_settings = dict(max_attempts=max_attempts, **CONFIGS[config], effort=getattr(llm, "effort", None),
                            max_tokens=getattr(llm, "max_tokens", None))
        row = dict(**tag, case=case["id"], config=config, rep=rep, model=model, library=lib,
                   category=case["category"], expect=case["expect"], prompt=prompt_id(config), **meta,
                   settings=run_settings,
                   **score(case, res),
                   requests=counter.requests - before[0],
                   input_tokens=llm.input_tokens - before[1],
                   output_tokens=llm.output_tokens - before[2],
                   **_timing(res, llm, before[3:]),
                   usd=None,
                   stl=_repo_relative(res.stl_path),
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        usd = price_usd(model, row["input_tokens"], row["output_tokens"])
        row["usd"] = round(usd, 4) if usd is not None else None
        spent += usd or 0
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        summary.ran += 1
        log(f"    -> {row['outcome']}, correct={row['correct']}, attempts={row['attempts']}, bbox_ok={row['bbox_ok']}"
            + (f", ${row['usd']:.4f} (total ${spent:.2f}" + (f" of ${max_usd:.2f})" if max_usd else ")")
               if usd is not None else ""))

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
    r.add_argument("--max-usd", type=float, help="stop before this model's results file (earlier runs included) "
                                                 "could cost more than this; needs the model in llm.PRICES")
    add_set_argument(r)
    rep = sub.add_parser("report", help="summarize results")
    rep.add_argument("files", nargs="*", type=Path, help="results files (default: all in eval/results, or the study's)")
    for p in (r, rep):
        p.add_argument("--study", help="keep results in eval/results/studies/<STUDY>/ (default: eval/results)")
    rep.add_argument("--judge-model", help="with --study: whose verdicts count for 'right shape' (default: the "
                                           "judge model in settings)")
    rep.add_argument("--threshold", type=float, help="judge pass threshold (default: the judge_threshold setting)")
    rep.add_argument("--organic-judge-model", help="with --study: another judge for the organic suite's verdicts "
                                                   "(the organic arms are then also shown under both judges)")
    st = sub.add_parser("study", help="run or check a study manifest (eval/studies/<name>.toml, see eval/study.py)")
    st.add_argument("action", choices=["run", "status"])
    st.add_argument("manifest", help="study name (eval/studies/<name>.toml) or path")
    st.add_argument("--run", nargs="+", dest="jobs", help="only these jobs (default: all, in file order)")
    st.add_argument("--budget", type=int, help="max LLM requests per job this time (default: the job's budget)")
    args = ap.parse_args(argv)

    if args.cmd == "study":
        from eval import study

        data = study.load_study(args.manifest)
        name = data["study"]["name"]
        if args.action == "run":
            unknown = set(args.jobs or []) - {j["name"] for j in data.get("run", [])}
            if unknown:
                raise SystemExit(f"No such job(s) in {name}: {sorted(unknown)}")
            for job in data.get("run", []):
                if args.jobs and job["name"] not in args.jobs:
                    continue
                s = study.run_job(name, job, args.budget)
                print(f"   ran {s.ran}, skipped {s.skipped}, {s.requests} requests"
                      + (f"; stopped: {s.stopped}" if s.stopped else ""))
                if s.stopped.startswith("error"):
                    sys.exit(2)
        print(study.status_table(data))
        return

    if args.cmd == "report":
        files = args.files or sorted(results_dir(args.study).glob("*.jsonl"))
        rows = [row for f in files for row in read_results(f)]
        if args.study:
            print(study_tables(args.study, rows, args.judge_model, args.threshold, args.organic_judge_model) + "\n")
        print(summarize(rows))
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
    path = results_dir(args.study) / f"{model}.jsonl"
    s = run(cases, configs, llm, model, ExampleRetriever.from_file(), path, args.budget, args.max_attempts,
            repeats=args.repeats, study=args.study, max_usd=args.max_usd)
    print(f"\nRan {s.ran}, skipped {s.skipped} already done, used {s.requests} requests. Results: {path}")
    if s.stopped:
        print("Stopped early: " + s.stopped)
        sys.exit(2 if s.stopped.startswith("error") else 0)


if __name__ == "__main__":
    main()
