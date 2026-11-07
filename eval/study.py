"""Study manifests: a whole benchmark round written down in one file, eval/studies/<name>.toml.

    python -m eval.bench study run report-v1 [--run gemini sonnet] [--budget N]
    python -m eval.bench study status report-v1

A manifest lists the round's build jobs, one [[run]] table each, run in file order:

    [[run]]
    name = "gemini"                       # job name, for --run and status
    model = "gemini-3.5-flash-lite"       # build model (download-only jobs: a name for the results file)
    suite = "main"                        # main | organic
    configs = ["rag-size", "norag-size"]  # bench configs
    repeats = 3
    budget = 400                          # max LLM requests per invocation (resume to continue)
    max_usd = 2.0                         # optional spending cap for this job's results file
    set = ["provider=gemini"]             # settings for this job only (like --set)
    cases = ["s01", "m02"]                # optional: only these cases

Comparisons to report are written down before the runs, one [[compare]] table each (arm = "model:config"):

    [[compare]]
    name = "RAG vs no RAG, Gemini"
    a = "gemini-3.5-flash-lite:rag-size"
    b = "gemini-3.5-flash-lite:norag-size"

`bench report --study <name>` prints the round's tables (eval/stats.py: rates per case with bootstrap CIs over
cases, paired comparisons with sign tests, timing), using the judge model's verdicts for "right shape".

Every job that sends LLM requests must set `provider` in `set`, so a job never runs on whatever the user's
settings.toml happens to say (no accidental paid requests). Results go to eval/results/studies/<name>/ (see
`bench --study`), and a job resumes where it stopped, like `bench run`.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from pathlib import Path

STUDIES_DIR = Path(__file__).parent / "studies"
JOB_KEYS = {"name", "model", "suite", "configs", "repeats", "budget", "max_usd", "set", "cases", "max_attempts"}


def load_study(name_or_path: str) -> dict:
    """The manifest as a dict, with its jobs checked. `name_or_path`: a name under eval/studies/ or a path."""
    from eval.bench import CONFIGS, SUITES

    path = Path(name_or_path)
    if not path.suffix:
        path = STUDIES_DIR / f"{name_or_path}.toml"
    data = tomllib.loads(path.read_text())
    data.setdefault("study", {}).setdefault("name", path.stem)
    names = set()
    for job in data.get("run", []):
        where = f"{path.name}, job {job.get('name', '?')!r}"
        unknown = set(job) - JOB_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        for key in ("name", "model", "configs"):
            if key not in job:
                raise ValueError(f"{where}: missing {key!r}")
        if job["name"] in names:
            raise ValueError(f"{where}: duplicate job name")
        names.add(job["name"])
        job.setdefault("suite", "main")
        job.setdefault("repeats", 1)
        job.setdefault("budget", 18)
        job.setdefault("set", [])
        if job["suite"] not in SUITES:
            raise ValueError(f"{where}: unknown suite {job['suite']!r}")
        bad = [c for c in job["configs"] if c not in CONFIGS]
        if bad:
            raise ValueError(f"{where}: unknown configs {bad}")
        if not download_only(job) and not any(s.split("=", 1)[0].strip() == "provider" for s in job["set"]):
            raise ValueError(f"{where}: set provider explicitly (e.g. set = [\"provider=gemini\"])")
    for cmp in data.get("compare", []):
        if set(cmp) != {"name", "a", "b"} or not all(":" in cmp[k] for k in ("a", "b")):
            raise ValueError(f"{path.name}: each [[compare]] needs name, a and b (\"model:config\"), got {cmp}")
    return data


def download_only(job: dict) -> bool:
    """A job whose configs send no LLM requests (Thingiverse downloads)."""
    from eval.bench import CONFIGS

    return all(CONFIGS[c].get("download") for c in job["configs"])


def job_cases(job: dict) -> list[dict]:
    from eval.bench import SUITES, load_cases

    cases = load_cases(SUITES[job["suite"]][0])
    return [c for c in cases if c["id"] in job["cases"]] if job.get("cases") else cases


def run_job(study: str, job: dict, budget: int | None = None, log=print):
    """Run one job (resumable). Returns bench.RunSummary."""
    from eval.bench import results_dir, run
    from stl_agent.config import clear_overrides, override_settings, reload_settings
    from stl_agent.llm import ScriptedLLM, make_llm
    from stl_agent.retrieval import ExampleRetriever

    clear_overrides()  # each job gets exactly its own settings
    override_settings(job["set"])  # (reloads settings)
    llm = ScriptedLLM([]) if download_only(job) else make_llm(model=job["model"])
    path = results_dir(study) / f"{job['model']}.jsonl"
    log(f"== {job['name']}: {job['model']}, {job['suite']} {' '.join(job['configs'])} x{job['repeats']} -> "
        f"{path.name}")
    try:
        return run(job_cases(job), job["configs"], llm, job["model"], ExampleRetriever.from_file(), path,
                   budget or job["budget"], job.get("max_attempts"), log=log, repeats=job["repeats"], study=study,
                   max_usd=job.get("max_usd"))
    finally:
        clear_overrides()
        reload_settings()  # clear_overrides alone leaves the job's settings cached


def job_status(study: str, job: dict) -> dict:
    """Rows done / expected and spend for one job."""
    from eval.bench import _lib_key, library_for, read_results, results_dir, spent_usd

    path = results_dir(study) / f"{job['model']}.jsonl"
    want = {(c["id"], k, n) for c in job_cases(job) for k in job["configs"] for n in range(job["repeats"])}
    have = {(r["case"], r["config"], r.get("rep", 0)) for r in read_results(path)
            if _lib_key(r["config"], r.get("library", "")) == _lib_key(r["config"], library_for(r["config"]))}
    return dict(name=job["name"], model=job["model"], done=len(want & have), expected=len(want),
                usd=spent_usd(path), max_usd=job.get("max_usd"))


def status_table(data: dict) -> str:
    study = data["study"]["name"]
    out = [f"## Study {study}\n", "| job | model | done | expected | spent | cap |", "|---|---|---|---|---|---|"]
    for job in data.get("run", []):
        s = job_status(study, job)
        cap = f"${s['max_usd']:.2f}" if s["max_usd"] is not None else "-"
        out.append(f"| {s['name']} | {s['model']} | {s['done']} | {s['expected']} | ${s['usd']:.2f} | {cap} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def suite_of(row: dict) -> str:
    return "organic" if row.get("category") == "organic" else "main"


def judge_verdicts(judge_rows: list[dict], threshold: float) -> dict[str, dict]:
    """part id -> {"pass": mean score over the judge's runs >= threshold, "seconds": mean judge seconds}, for
    judged benchmark parts with a usable score."""
    by_part = defaultdict(list)
    for r in judge_rows:
        if r.get("kind") == "bench" and r.get("score") is not None:
            by_part[r["part"]].append(r)
    return {pid: {"pass": sum(r["score"] for r in rs) / len(rs) >= threshold,
                  "seconds": sum(r.get("seconds") or 0 for r in rs) / len(rs) if any("seconds" in r for r in rs) else None}
            for pid, rs in by_part.items()}


def shape_ok(row: dict, verdicts: dict[str, dict]) -> bool | None:
    """Right size and right shape: the benchmark's check passed and the judge passed the part. Organic suite (no
    size to check): any part that came out and the judge passes, watertight or not, since the app hands those
    parts over too (as in eval/results/organic-sdf.md). A run with no part counts as a failure; a part the judge
    hasn't scored is left out (None)."""
    from eval.bench import part_id

    if suite_of(row) == "organic":
        if row["outcome"] not in ("built", "best_effort"):
            return False
    elif not row["correct"] or row["outcome"] != "built":
        return False if row.get("expect") == "build" else bool(row["correct"])
    v = verdicts.get(part_id(row))
    return None if v is None else v["pass"]


def arms(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """(model, suite, config) -> rows."""
    out = defaultdict(list)
    for r in rows:
        out[(r["model"], suite_of(r), r["config"])].append(r)
    return dict(sorted(out.items()))


def study_report(rows: list[dict], verdicts: dict[str, dict] | None = None, compare: list[dict] | None = None,
                 judge_name: str = "", organic_verdicts: dict[str, dict] | None = None,
                 organic_judge_name: str = "") -> str:
    """Markdown tables for a study: results per arm with bootstrap CIs over cases, timing, and the comparisons
    named in the manifest. `organic_verdicts`: another judge's verdicts used for the organic suite (when it agrees
    better with the hand grades on organic parts); a table then shows the organic arms under both judges."""
    from eval import stats
    from eval.bench import part_id

    verdicts = verdicts or {}
    main_verdicts = verdicts
    if organic_verdicts is not None:
        verdicts = {part_id(r): v for r in rows
                    if (v := (organic_verdicts if suite_of(r) == "organic" else main_verdicts).get(part_id(r)))}
        judge_name = f"{judge_name}; organic suite: {organic_judge_name}"
    judged = bool(verdicts)
    out = ["## Results per model and config\n",
           "Rates are the mean over cases of each case's rate over its repeats; [brackets] are 95% bootstrap "
           "intervals over cases. Main suite: correct = built at the expected size (or declined when it should be). "
           "Organic suite: correct = built as a valid mesh (no expected size)."
           + (f" Right shape: correct, and the visual judge ({judge_name}) passes the part; for the organic suite, any "
              "part that came out (valid mesh or not) that the judge passes." if judged else "") + "\n"]
    head = ["model", "suite", "config", "cases", "runs", "correct"] + (["right size + shape", "judged"] if judged else []) + \
        ["built", "first try", "requests/run", "$/run"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (model, suite, config), rs in arms(rows).items():
        n_cases = len({r["case"] for r in rs})
        correct = stats.bootstrap_mean(stats.per_case(rs, lambda r: r["correct"]).values())
        cells = [model, suite, config, str(n_cases), str(len(rs)), stats.fmt_rate(*correct)]
        if judged:
            shape = stats.bootstrap_mean(stats.per_case(rs, lambda r: shape_ok(r, verdicts)).values())
            n_judged = sum(1 for r in rs if part_id(r) in verdicts)
            cells += [stats.fmt_rate(*shape), str(n_judged)]
        usd = [r["usd"] for r in rs if r.get("usd") is not None]
        cells += [f"{sum(bool(r['built']) for r in rs) / len(rs):.0%}", f"{sum(bool(r['first_try']) for r in rs) / len(rs):.0%}",
                  f"{sum(r['requests'] for r in rs) / len(rs):.2f}", f"${sum(usd) / len(usd):.4f}" if usd else "-"]
        out.append("| " + " | ".join(cells) + " |")

    if organic_verdicts is not None:
        out += [f"\n### Organic suite under both judges\n",
                "Right shape (any part that came out, passed by the judge), mean over cases [95% interval].\n",
                f"| model | config | {organic_judge_name} | {main_judge_label(judge_name)} |", "|---|---|---|---|"]
        for (model, suite, config), rs in arms(rows).items():
            if suite != "organic":
                continue
            cells = [stats.fmt_rate(*stats.bootstrap_mean(stats.per_case(rs, lambda r, v=v: shape_ok(r, v)).values()))
                     for v in (organic_verdicts, main_verdicts)]
            out.append(f"| {model} | {config} | {cells[0]} | {cells[1]} |")

    out += ["\n## Tokens and cost\n",
            "Per run, all attempts included; output tokens include the model's thinking. Per correct part: everything "
            "spent on the arm divided by its correct runs, so failed and repaired runs count. Each model family splits "
            "text into tokens differently, so compare token counts within a family, and cost or time across families.\n",
            "| model | suite | config | input tokens/run | output tokens/run | tokens per correct part | $ per correct part |",
            "|---|---|---|---|---|---|---|"]
    for (model, suite, config), rs in arms(rows).items():
        n_ok = sum(bool(r["correct"]) for r in rs)
        tin, tout = sum(r.get("input_tokens", 0) for r in rs), sum(r.get("output_tokens", 0) for r in rs)
        usd = [r["usd"] for r in rs if r.get("usd") is not None]
        per_ok = f"{(tin + tout) / n_ok:,.0f}" if n_ok and tin + tout else "-"
        usd_ok = f"${sum(usd) / n_ok:.4f}" if n_ok and len(usd) == len(rs) else "-"
        out.append(f"| {model} | {suite} | {config} | {tin / len(rs):,.0f} | {tout / len(rs):,.0f} | {per_ok} | {usd_ok} |")

    out += ["\n## Time\n",
            "Seconds, median (90th percentile). Build: wall clock for one run. Request: time inside one LLM request. "
            "Waiting: rate pacing and retry back-off per run. Sandbox: running and checking scripts per run. "
            "Judge: one judge request per part. End to end: build + judge for a judged part.\n",
            "| model | suite | config | build | request | waiting | sandbox | output tokens/s | judge | end to end |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    q = lambda xs: "{} ({})".format(*(stats.fmt_s(v) for v in stats.quantiles(xs)))
    for (model, suite, config), rs in arms(rows).items():
        timed = [r for r in rs if "llm_seconds" in r]
        per_request = [r["llm_seconds"] / r["requests"] for r in timed if r["requests"]]
        llm_s = sum(r["llm_seconds"] for r in timed)
        tps = f"{sum(r['output_tokens'] for r in timed) / llm_s:.0f}" if llm_s else "-"
        judge_s = [verdicts[part_id(r)]["seconds"] for r in rs if part_id(r) in verdicts]
        e2e = [r["seconds"] + verdicts[part_id(r)]["seconds"] for r in rs
               if part_id(r) in verdicts and verdicts[part_id(r)]["seconds"] is not None]
        out.append(f"| {model} | {suite} | {config} | {q(r['seconds'] for r in rs)} | {q(per_request)} | "
                   f"{q(r['wait_seconds'] for r in timed)} | {q(r['sandbox_seconds'] for r in timed)} | {tps} | "
                   f"{q(judge_s)} | {q(e2e)} |")

    if compare:
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[f"{r['model']}:{r['config']}"].append(r)
        metrics = [("correct", lambda r: r["correct"])] + ([("right size + shape", lambda r: shape_ok(r, verdicts))]
                                                          if judged else [])
        out += ["\n## Comparisons (named in the manifest before the runs)\n",
                "Paired by case: the mean per-case difference a - b with a 95% bootstrap interval, the cases where a "
                "or b did better, and an exact sign test on those cases (ties left out).\n",
                "| comparison | metric | cases | a | b | a - b | a better / b better / tied | p |",
                "|---|---|---|---|---|---|---|---|"]
        for cmp in compare:
            ra, rb = by_arm.get(cmp["a"], []), by_arm.get(cmp["b"], [])
            for metric, fn in metrics:
                p = stats.paired(stats.per_case(ra, fn), stats.per_case(rb, fn))
                if p["n"] == 0:
                    out.append(f"| {cmp['name']} | {metric} | 0 | - | - | - | - | - |")
                    continue
                out.append(f"| {cmp['name']} | {metric} | {p['n']} | {p['a']:.0%} | {p['b']:.0%} | {stats.fmt_diff(p)} | "
                           f"{p['better']} / {p['worse']} / {p['tied']} | {p['p']:.3f} |")
        out.append("\na = " + "; ".join(f"{c['name']}: {c['a']} vs {c['b']}" for c in compare))
    return "\n".join(out)


def main_judge_label(judge_name: str) -> str:
    """The main judge's name from a combined label ("A; organic suite: B" -> "A")."""
    return judge_name.split(";")[0]
