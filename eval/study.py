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

Every job that sends LLM requests must set `provider` in `set`, so a job never runs on whatever the user's
settings.toml happens to say (no accidental paid requests). Results go to eval/results/studies/<name>/ (see
`bench --study`), and a job resumes where it stopped, like `bench run`.
"""

from __future__ import annotations

import tomllib
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
