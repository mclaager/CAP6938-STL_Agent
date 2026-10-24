"""LLM-as-judge evaluation (issue #8): render benchmark parts and score them with a multimodal model.

    python -m eval.judge run --gen-model gemini-3.5-flash-lite --config rag norag   # judge benchmark parts
    python -m eval.judge calibrate      # judge reference parts (+ your labels) and pick a pass threshold
    python -m eval.judge report         # pass rates per config/category, judge vs bounding-box check
    python -m eval.judge sheet          # eval/renders/index.html: browse renders, grade parts

Each judged part is one request. Results are appended to eval/results/judge/<judge-model>.jsonl, one row
per (part, judge, prompt, render settings); a run skips rows already there, so changing the judge model
(--judge-model / CAD_JUDGE_MODEL), prompt (--prompt) or resolution (--px) adds a new study next to the
old one. Renders are saved once under eval/renders/<render version>-<px>px/ (committed, ~35 KB each).

Parts come from benchmark runs (eval/bench.py), whose STLs are under outputs/bench/ (gitignored, local).
"""

from __future__ import annotations

import argparse
import html
import json
import tempfile
import time
from pathlib import Path

from eval.bench import RESULTS_DIR, CountingLLM, _lib_key, library_id, load_cases, read_results
from stl_agent.config import JUDGE_THRESHOLD, OUTPUT_DIR, ROOT
from stl_agent.render import RENDER_VERSION, png_bytes, render_grid

EVAL_DIR = Path(__file__).parent
RENDERS_DIR = EVAL_DIR / "renders"
JUDGE_DIR = RESULTS_DIR / "judge"
LABELS_PATH = EVAL_DIR / "judge_labels.json"
DEFAULT_PX = 256
MISMATCH_OFFSET = 7  # reference i is also judged against the request of reference i+7 (should fail)
# Near misses: a reference judged against the request of a similar-looking part (should fail).
# The offset pairs above are all clearly different objects; these test whether the judge notices
# a missing or wrong feature. (part, request it is judged against)
NEAR_MISSES = [("m14", "h13"), ("h13", "m14"), ("c02", "h13"), ("m09", "m14"), ("m10", "m13"), ("m13", "m10"),
               ("m15", "m13"), ("h04", "h12"), ("h12", "h04"), ("m12", "h04"), ("h11", "c02"), ("c01", "h08"),
               ("h08", "c01"), ("h08", "m09")]


# ---------------------------------------------------------------------------
# Finding parts and their renders
# ---------------------------------------------------------------------------
def part_id(row: dict) -> str:
    """Stable id of a benchmark part, e.g. "gemini-3.5-flash-lite/lib26-cdabac/rag/m04/rep0"."""
    return f"{row['model']}/{row.get('library') or 'lib9'}/{row['config']}/{row['case']}/rep{row.get('rep', 0)}"


def render_tag(px: int) -> str:
    return f"{RENDER_VERSION}-{px}px"


def render_path(pid: str, px: int) -> Path:
    """eval/renders/<render version>-<px>px/<part id path>.png, one folder per render setting."""
    head, case, rep = pid.rsplit("/", 2)
    return RENDERS_DIR / render_tag(px) / head / f"{case}-{rep}.png"


def locate_stl(row: dict) -> Path | None:
    """The part a benchmark row produced: its recorded `stl`, else the final attempt in its run folder."""
    if row.get("stl") and (ROOT / row["stl"]).exists():
        return ROOT / row["stl"]
    run_dir = next((d for d in run_dirs(row) if (d / "report.json").exists()), None)
    if run_dir is None:
        return None
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("stl_path"):
        return Path(report["stl_path"]) if Path(report["stl_path"]).exists() else None
    best = best_attempt(report.get("attempt_log") or [])
    path = run_dir / f"attempt_{best}" / "model.stl" if best else None
    return path if path and path.exists() else None


def run_dirs(row: dict) -> list[Path]:
    """Where bench.run saved this row's run. Rows from before library versions were recorded used
    outputs/bench/<model>/<config>/<case>/, with repeats 1+ in rep<k>/ below it."""
    dirs = [OUTPUT_DIR / "bench" / part_id(row)]
    if not row.get("library"):
        legacy = OUTPUT_DIR / "bench" / row["model"] / row["config"] / row["case"]
        dirs.append(legacy if row.get("rep", 0) == 0 else legacy / f"rep{row['rep']}")
    return dirs


def best_attempt(attempt_log: list[dict]) -> int | None:
    """Which attempt generate_cad kept: the first valid one, else the best-effort one (same rule)."""
    best = None
    for rec in attempt_log:
        val = rec.get("validation")
        if not val:
            continue
        if best is None or (val["valid"] and not best[1]["valid"]) or (val["watertight"] and not best[1]["watertight"]):
            best = (rec["attempt"], val)
        if val["valid"]:
            break
    return best[0] if best else None


def ensure_render(stl: Path, out: Path, px: int) -> bytes:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png_bytes(render_grid(str(stl), px)))
    return out.read_bytes()


def bench_rows(gen_model: str, configs: list[str] | None, case_ids: list[str] | None, library: str) -> list[dict]:
    """Latest benchmark row per (case, config, rep) for this library (no-RAG rows from any library)."""
    latest = {}
    for r in read_results(RESULTS_DIR / f"{gen_model}.jsonl"):
        if configs and r["config"] not in configs or case_ids and r["case"] not in case_ids:
            continue
        if _lib_key(r["config"], r.get("library") or "lib9") not in ("any", library):
            continue
        latest[(r["case"], r["config"], r.get("rep", 0))] = r
    return list(latest.values())


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
class Session:
    """One judge + one results file; skips (part, judge, prompt, render) rows already recorded."""

    def __init__(self, judge, judge_name: str, prompt: str, px: int, budget: int, log=print):
        self.judge, self.prompt, self.px, self.budget, self.log = judge, prompt, px, budget, log
        self.path = JUDGE_DIR / f"{judge_name}.jsonl"
        self.done = {(r["part"], r["judge"], r["prompt"], r["render"]) for r in read_results(self.path)}
        self.requests, self.stopped = 0, ""

    def is_done(self, pid: str) -> bool:
        return (pid, self.judge.name, self.prompt, render_tag(self.px)) in self.done

    def judge_one(self, pid: str, request: str, image: bytes, facts: dict, **extra) -> dict | None:
        """Judge and append a row. Returns None (and sets `stopped`) at the budget or on an API error."""
        if self.requests >= self.budget:
            self.stopped = f"budget: {self.requests}/{self.budget} requests used; run again to continue"
            return None
        llm = getattr(self.judge, "llm", None)
        before = (getattr(llm, "input_tokens", 0), getattr(llm, "output_tokens", 0))
        try:
            self.requests += 1
            j = self.judge.judge(request, image, facts)
        except Exception as e:  # quota exhausted, network: stop without recording
            self.stopped = f"error, stopping: {type(e).__name__}: {e}"
            return None
        row = dict(part=pid, judge=self.judge.name, prompt=self.prompt, render=render_tag(self.px), **extra,
                   request=request, facts=facts, score=j.score, raw_score=j.raw_score, matches=j.matches,
                   problems=j.problems, error=j.error,
                   input_tokens=getattr(llm, "input_tokens", 0) - before[0],
                   output_tokens=getattr(llm, "output_tokens", 0) - before[1],
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if j.error:
            row["reply"] = j.reply[:2000]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add((pid, self.judge.name, self.prompt, render_tag(self.px)))
        self.log(f"  {pid}: score {j.score if j.score is not None else 'ERROR ' + str(j.error)}"
                 + (f" | {'; '.join(j.problems)[:120]}" if j.problems else ""))
        return row


def judge_bench_row(s: Session, row: dict, case: dict) -> str:
    """Judge one benchmark part. Returns what happened: judged / done / no_part / missing_stl / stopped."""
    pid = part_id(row)
    if row["outcome"] not in ("built", "best_effort"):
        return "no_part"  # declined or failed: nothing to look at
    if s.is_done(pid):
        return "done"
    stl = locate_stl(row)
    if stl is None:
        return "missing_stl"
    out = render_path(pid, s.px)
    image = ensure_render(stl, out, s.px)
    extra = dict(kind="bench", case=row["case"], config=row["config"], rep=row.get("rep", 0), gen_model=row["model"],
                 library=row.get("library"), category=row["category"], bench_outcome=row["outcome"],
                 bench_correct=row["correct"], render_file=str(out.relative_to(EVAL_DIR)))
    return "judged" if s.judge_one(pid, case["request"], image, {"bbox_mm": row.get("bbox_mm")}, **extra) else "stopped"


def cmd_run(args, s: Session) -> None:
    cases = {c["id"]: c for c in load_cases()}
    rows = bench_rows(args.gen_model, args.config, args.cases, args.library or library_id())
    if args.reps is not None:
        rows = [r for r in rows if r.get("rep", 0) in args.reps]
    if not rows:
        print(f"No benchmark rows for {args.gen_model} (eval/results/{args.gen_model}.jsonl).")
        return
    tally = {}
    for row in sorted(rows, key=lambda r: (r.get("rep", 0), r["config"], r["case"])):
        what = judge_bench_row(s, row, cases[row["case"]])
        tally[what] = tally.get(what, 0) + 1
        if what == "stopped":
            break
    print(f"\n{tally} | {s.requests} requests | results: {s.path.relative_to(ROOT)}")
    if s.stopped:
        print("Stopped: " + s.stopped)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_items(px: int) -> list[dict]:
    """Reference parts against their own request (should pass), an unrelated case's request and a
    similar part's request (both should fail)."""
    import cadquery as cq
    import trimesh

    from eval.references import REFS

    cases = {c["id"]: c for c in load_cases()}
    ids = sorted(REFS)
    items = []
    for i, cid in enumerate(ids):
        out = RENDERS_DIR / render_tag(px) / "references" / f"{cid}.png"
        with tempfile.TemporaryDirectory() as tmp:
            stl = Path(tmp) / f"{cid}.stl"
            cq.exporters.export(REFS[cid](), str(stl))
            image = ensure_render(stl, out, px)
            bbox = [round(float(x), 2) for x in trimesh.load(str(stl), force="mesh").extents]
        other = ids[(i + MISMATCH_OFFSET) % len(ids)]
        common = dict(case=cid, render_file=str(out.relative_to(EVAL_DIR)), image=image, facts={"bbox_mm": bbox})
        items.append(dict(common, pid=f"references/{cid}", kind="reference", request=cases[cid]["request"], label=True))
        items.append(dict(common, pid=f"references/{cid}@{other}", kind="mismatch", request=cases[other]["request"],
                          label=False, asked_case=other))
        for part, asked in NEAR_MISSES:
            if part == cid:
                items.append(dict(common, pid=f"references/{cid}@{asked}", kind="near_miss",
                                  request=cases[asked]["request"], label=False, asked_case=asked))
    return items


# Hand-label grades, the same four bands the judge prompt scores on (1-10 scale). The top two pass.
GRADES = ["match", "minor", "feature_wrong", "different"]
GRADE_NAMES = {"match": "match", "minor": "minor deviation", "feature_wrong": "feature wrong", "different": "different object"}
PASS_GRADES = {"match", "minor"}


def band(raw_score: float | None) -> str | None:
    """The prompt's band for an LLM judge score: 9-10 match, 7-8 minor, 4-6 feature wrong, 1-3 different."""
    if raw_score is None:
        return None
    return "match" if raw_score >= 9 else "minor" if raw_score >= 7 else "feature_wrong" if raw_score >= 4 else "different"


def label_pass(label: dict) -> bool | None:
    """Pass/fail derived from a hand label's grade (match / minor pass). None when it has no valid grade."""
    return label["grade"] in PASS_GRADES if label.get("grade") in GRADES else None


def band_agreement(pairs: list[tuple[str, str]]) -> dict:
    """(judge band, human grade) pairs -> exact / within-one agreement, lenient / harsh counts, 4x4 counts.
    Lenient: the judge's band is better than the human grade (e.g. 'match' for a 'feature_wrong' part)."""
    idx = {g: i for i, g in enumerate(GRADES)}
    diffs = [idx[h] - idx[j] for j, h in pairs]  # > 0: judge rated it better than the human did
    matrix = {(j, h): sum(1 for p in pairs if p == (j, h)) for j in GRADES for h in GRADES}
    n = len(pairs) or 1
    return dict(n=len(pairs), exact=sum(d == 0 for d in diffs) / n, within_one=sum(abs(d) <= 1 for d in diffs) / n,
                lenient=sum(d > 0 for d in diffs), harsh=sum(d < 0 for d in diffs), matrix=matrix)


def load_labels() -> dict:
    labels = json.loads(LABELS_PATH.read_text()).get("labels", {}) if LABELS_PATH.exists() else {}
    ungraded = [pid for pid, label in labels.items() if label_pass(label) is None]
    if ungraded:
        print(f"Note: {len(ungraded)} label(s) without a valid grade ({', '.join(GRADES)}) are ignored: "
              + ", ".join(ungraded[:5]) + (" ..." if len(ungraded) > 5 else ""))
    return labels


def cmd_calibrate(args, s: Session) -> None:
    for it in calibration_items(s.px):
        if s.is_done(it["pid"]):
            continue
        extra = {k: it[k] for k in ("kind", "case", "render_file", "label") if k in it}
        if "asked_case" in it:
            extra["asked_case"] = it["asked_case"]
        if not s.judge_one(it["pid"], it["request"], it["image"], it["facts"], **extra):
            break
    labels = load_labels()
    if labels and not s.stopped:  # labelled benchmark parts that haven't been judged yet
        cases = {c["id"]: c for c in load_cases()}
        for pid in labels:
            if s.stopped or s.is_done(pid) or label_pass(labels[pid]) is None:
                continue
            gen_model, library, config, case, rep = pid.split("/")
            rows = [r for r in bench_rows(gen_model, [config], [case], library) if f"rep{r.get('rep', 0)}" == rep]
            if rows:
                judge_bench_row(s, rows[0], cases[case])
    if s.stopped:
        print("Stopped: " + s.stopped)
    print(calibration_report(s, labels))


def calibration_report(s: Session, labels: dict) -> str:
    rows = [r for r in read_results(s.path)
            if r["judge"] == s.judge.name and r["prompt"] == s.prompt and r["render"] == render_tag(s.px)]
    data = []  # (score, label, source)
    for r in rows:
        if r["score"] is None:
            continue
        if r.get("kind") in ("reference", "mismatch", "near_miss"):
            data.append((r["score"], r["label"], r["kind"]))
        elif r["part"] in labels and label_pass(labels[r["part"]]) is not None:
            data.append((r["score"], label_pass(labels[r["part"]]), "label"))
    if not data:
        return "No calibration results yet."
    errors = sum(1 for r in rows if r["score"] is None)
    out = [f"## Calibration: {s.judge.name}, prompt {s.prompt}, render {render_tag(s.px)}\n",
           f"{len(data)} judged items ({sum(1 for d in data if d[1])} should pass, "
           f"{sum(1 for d in data if not d[1])} should fail){f', {errors} unusable replies' if errors else ''}. "
           f"AUC {auc(data):.2f}.\n",
           "| threshold | accuracy | balanced accuracy | references passed | unrelated failed | near misses failed | labels right |",
           "|---|---|---|---|---|---|---|"]
    best = max(threshold_table(data), key=lambda t: (round(t["balanced"], 6), -abs(t["threshold"] - 0.5)))
    for t in threshold_table(data):
        mark = " **(best)**" if t is best or t["threshold"] == best["threshold"] else ""
        out.append(f"| {t['threshold']:.2f}{mark} | {t['accuracy']:.0%} | {t['balanced']:.0%} | "
                   f"{t['reference']} | {t['mismatch']} | {t['near_miss']} | {t['label']} |")
    out.append(f"\nBest threshold {best['threshold']:.2f}: set CAD_JUDGE_THRESHOLD={best['threshold']:.2f} "
               "(or update the default in stl_agent/config.py).")
    pairs = [(band(r["raw_score"]), labels[r["part"]]["grade"]) for r in rows
             if r["score"] is not None and r["part"] in labels and label_pass(labels[r["part"]]) is not None]
    if pairs:
        out.append("\n" + agreement_table(band_agreement(pairs)))
    return "\n".join(out)


def agreement_table(a: dict) -> str:
    out = [f"### Band agreement with your graded labels ({a['n']} parts)\n",
           f"Exact band {a['exact']:.0%}, within one band {a['within_one']:.0%}. "
           f"Judge more lenient than you on {a['lenient']}, harsher on {a['harsh']}.\n",
           "| judge band \\ your grade | " + " | ".join(GRADE_NAMES[g] for g in GRADES) + " |",
           "|---|" + "---|" * len(GRADES)]
    for j in GRADES:
        out.append(f"| {GRADE_NAMES[j]} | " + " | ".join(str(a["matrix"][(j, h)]) for h in GRADES) + " |")
    return "\n".join(out)


def threshold_table(data: list[tuple]) -> list[dict]:
    """Accuracy at each candidate threshold (a part passes when score >= threshold). Candidates sit
    halfway between neighbouring scores, so the chosen threshold isn't on top of an observed score."""
    scores = sorted({d[0] for d in data})
    table = []
    for t in [round((a + b) / 2, 4) for a, b in zip(scores, scores[1:])] or scores:
        right = lambda src: [(d[0] >= t) == d[1] for d in data if d[2] == src]
        pos = [d[0] >= t for d in data if d[1]]
        neg = [d[0] < t for d in data if not d[1]]
        frac = lambda xs: f"{sum(xs)}/{len(xs)}" if xs else "–"
        table.append(dict(threshold=t, accuracy=sum((d[0] >= t) == d[1] for d in data) / len(data),
                          balanced=((sum(pos) / len(pos) if pos else 1) + (sum(neg) / len(neg) if neg else 1)) / 2,
                          reference=frac(right("reference")), mismatch=frac(right("mismatch")),
                          near_miss=frac(right("near_miss")), label=frac(right("label"))))
    return table


def auc(data: list[tuple]) -> float:
    pos = [d[0] for d in data if d[1]]
    neg = [d[0] for d in data if not d[1]]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Report and review sheet
# ---------------------------------------------------------------------------
def judged_bench(path: Path, judge_name: str | None = None) -> list[dict]:
    rows = [r for r in read_results(path) if r.get("kind") == "bench" and (judge_name is None or r["judge"] == judge_name)]
    latest = {}
    for r in rows:
        latest[(r["part"], r["judge"], r["prompt"], r["render"])] = r
    return list(latest.values())


def summarize(rows: list[dict], threshold: float) -> str:
    if not rows:
        return "No judged benchmark parts yet."
    out = []
    groups = sorted({(r["judge"], r["prompt"], r["render"], r["gen_model"]) for r in rows})
    for judge, prompt, render, gen in groups:
        rs = [r for r in rows if (r["judge"], r["prompt"], r["render"], r["gen_model"]) == (judge, prompt, render, gen)]
        out.append(f"## Parts by {gen}, judged by {judge} (prompt {prompt}, render {render}, pass at score >= {threshold:.2f})\n")
        out.append("| config / category | judged | judge pass | mean score | bbox correct | bbox correct + judge pass |")
        out.append("|---|---|---|---|---|---|")
        cats = ["all", "simple", "medium", "hard", "proposal", "casual"]
        for config in sorted({r["config"] for r in rs}):
            for cat in cats:
                xs = [r for r in rs if r["config"] == config and (cat == "all" or r["category"] == cat) and r["score"] is not None]
                if not xs:
                    continue
                n = len(xs)
                passed = [r for r in xs if r["score"] >= threshold]
                both = [r for r in passed if r["bench_correct"]]
                label = f"**{config}**" if cat == "all" else f"{config} / {cat}"
                out.append(f"| {label} | {n} | {len(passed) / n:.0%} | {sum(r['score'] for r in xs) / n:.2f} | "
                           f"{sum(bool(r['bench_correct']) for r in xs) / n:.0%} | {len(both) / n:.0%} |")
        out.append("\n| config | " + " | ".join(GRADE_NAMES[g] for g in GRADES) + " |\n|---|" + "---|" * len(GRADES))
        for config in sorted({r["config"] for r in rs}):
            bands = [band(r["raw_score"]) for r in rs if r["config"] == config and r["score"] is not None]
            out.append(f"| {config} | " + " | ".join(str(bands.count(g)) for g in GRADES) + " |")
        usable = [r for r in rs if r["score"] is not None]
        cell = lambda c, p: sum(1 for r in usable if bool(r["bench_correct"]) == c and (r["score"] >= threshold) == p)
        out.append("\n| | judge pass | judge fail |\n|---|---|---|")
        out.append(f"| bbox correct | {cell(True, True)} | {cell(True, False)} (right size, wrong shape?) |")
        out.append(f"| bbox wrong | {cell(False, True)} (looks right, size off?) | {cell(False, False)} |")
        errors = len(rs) - len(usable)
        if errors:
            out.append(f"\n{errors} unusable judge replies left out.")
        out.append("")
    return "\n".join(out)


SHEET_CSS = """
body{font:14px system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}
h1{font-size:20px} h2{font-size:16px;margin-top:28px}
.bar{position:sticky;top:0;background:#f6f6f4;padding:8px 0;display:flex;gap:8px;flex-wrap:wrap;z-index:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
.card img{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in}
.card.big{grid-column:1/-1} .card.big img{max-width:768px}
.id{font:12px ui-monospace,monospace;color:#555;word-break:break-all}
.req{margin:6px 0}.meta{font-size:12px;color:#444}.prob{font-size:12px;color:#a33;margin:4px 0 0 16px;padding:0}
.lbl button{margin:6px 4px 0 0;font-size:12px}.lbl .on{background:#246;color:#fff}
.lbl input{width:100%;margin-top:4px;font-size:12px;box-sizing:border-box}
textarea{width:100%;height:120px;font:12px ui-monospace,monospace}
"""

SHEET_JS = """
const KEY = 'stl-agent-judge-labels';
const labels = JSON.parse(document.getElementById('labels').textContent);
try { Object.assign(labels, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}  // in-progress labels
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function grade(pid, g){ const l = labels[pid] || {};
  if (l.grade === g) delete l.grade; else l.grade = g;
  if (l.grade || l.why) labels[pid] = l; else delete labels[pid]; save(); paint(); }
function why(pid, text){ const l = labels[pid] || {}; l.why = text;
  if (l.grade || l.why) labels[pid] = l; else delete labels[pid]; save(); paint(); }
function paint(){ document.querySelectorAll('.lbl').forEach(d => { const l = labels[d.dataset.pid] || {};
  d.querySelectorAll('button').forEach(b => b.classList.toggle('on', l.grade === b.dataset.g));
  const w = d.querySelector('input'); if (document.activeElement !== w) w.value = l.why || ''; });
  document.getElementById('out').value = JSON.stringify({labels}, null, 1);
  const graded = Object.values(labels).filter(l => l.grade).length;
  document.getElementById('count').textContent = graded + ' graded'; }
function copyLabels(){ navigator.clipboard.writeText(document.getElementById('out').value); }
function filt(q){ q = q.toLowerCase(); document.querySelectorAll('.card').forEach(c =>
  c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none'); }
document.querySelectorAll('.card img').forEach(i => i.onclick = () => i.parentElement.classList.toggle('big'));
paint();
"""


def write_sheet(path: Path, threshold: float, render: str | None = None) -> Path:
    """Static review page next to the renders: every judged part and reference, with label buttons.
    `render`: only rows with this render setting (e.g. "v2-256px"); None shows all."""
    cases = {c["id"]: c for c in load_cases()}
    rows = []
    for f in sorted(JUDGE_DIR.glob("*.jsonl")):
        rows += read_results(f)
    latest = {}
    for r in rows:
        if render is None or r["render"] == render:
            latest[(r["part"], r["judge"], r["prompt"], r["render"])] = r
    by_render = {}
    for r in latest.values():
        by_render.setdefault(r["render_file"], []).append(r)
    labels = load_labels()

    def card(render_file: str, rs: list[dict], labellable: bool) -> str:
        r0 = rs[0]
        case = cases.get(r0["case"], {})
        meta = (f"bench: {r0['bench_outcome']}, bbox {'ok' if r0['bench_correct'] else 'wrong'}"
                if r0.get("kind") == "bench" else "reference solution")
        verdicts = []
        for r in sorted(rs, key=lambda r: (r["part"], r["judge"])):
            asked = f" (asked: {r['asked_case']})" if r.get("asked_case") else ""
            score = ("error" if r["score"] is None else
                     f"{r['score']:.2f} {'pass' if r['score'] >= threshold else 'FAIL'} ({GRADE_NAMES[band(r['raw_score'])]})")
            probs = "".join(f"<li>{html.escape(p)}</li>" for p in r["problems"])
            verdicts.append(f"<div class=meta>{html.escape(r['judge'])} {r['prompt']}{asked}: <b>{score}</b></div>"
                            + (f"<ul class=prob>{probs}</ul>" if probs else ""))
        pid = html.escape(r0["part"])
        lbl = (f"<div class=lbl data-pid='{pid}'>"
               + "".join(f"<button data-g={g} onclick=\"grade('{pid}','{g}')\">{GRADE_NAMES[g]}</button>" for g in GRADES)
               + f"<input placeholder='why (optional)' oninput=\"why('{pid}',this.value)\"></div>"
               if labellable else "")
        return (f"<div class=card><img loading=lazy src='{html.escape(str(Path(render_file).relative_to('renders')))}'>"
                f"<div class=id>{html.escape(r0['part'] if r0.get('kind') == 'bench' else 'references/' + r0['case'])}</div>"
                f"<div class=req>{html.escape(case.get('request', ''))}</div><div class=meta>{meta}</div>"
                + "".join(verdicts) + lbl + "</div>")

    sections = {}
    for render_file, rs in sorted(by_render.items()):
        r0 = rs[0]
        if r0.get("kind") != "bench":
            key = "Reference solutions (calibration)"
        else:  # no-RAG parts don't depend on the example library, so they share one section
            lib = r0.get("library") or "lib9"
            key = f"{r0['gen_model']} / {'any library' if _lib_key(r0['config'], lib) == 'any' else lib} / {r0['config']}"
        sections.setdefault(key, []).append(card(render_file, rs, r0.get("kind") == "bench"))
    body = "".join(f"<h2>{html.escape(k)} ({len(v)})</h2><div class=grid>{''.join(v)}</div>" for k, v in sections.items())
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Judge renders</title><style>{SHEET_CSS}</style></head><body>
<h1>Judge renders</h1>
<p>Generated by <code>python -m eval.judge sheet</code>. Renders: {html.escape(render or "all versions")}.
Pass threshold {threshold:.2f}. Click an image to enlarge.
To label: grade each part on the judge's scale (match and minor deviation pass; feature wrong and different object
fail), optionally say why, then copy the JSON into <code>eval/judge_labels.json</code>. Grades are kept in this
browser until then, so reloading or regenerating the page doesn't lose them.</p>
<div class=bar><input placeholder="filter (case, config, text...)" oninput="filt(this.value)">
<button onclick=copyLabels()>Copy labels JSON</button><span id=count></span></div>
{body}
<h2>Labels JSON</h2><textarea id=out readonly></textarea>
<script type=application/json id=labels>{json.dumps(labels).replace('</', '<\\/')}</script>
<script>{SHEET_JS}</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    from stl_agent.judge import DEFAULT_PROMPT, PROMPTS, judge_model, make_judge

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def judge_args(p):
        p.add_argument("--judge-model", help="judge model (default: CAD_JUDGE_MODEL / provider's judge default)")
        p.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
        p.add_argument("--px", type=int, default=DEFAULT_PX, help="pixels per view (the render is 2*px square)")

    r = sub.add_parser("run", help="judge benchmark parts")
    r.add_argument("--gen-model", required=True, help="model whose benchmark parts to judge")
    r.add_argument("--config", nargs="+", help="benchmark configs (default: all)")
    r.add_argument("--cases", nargs="+", help="case ids (default: all)")
    r.add_argument("--library", help="example-library id for RAG rows (default: current library)")
    r.add_argument("--reps", nargs="+", type=int, help="only these benchmark repeats, e.g. --reps 0 (default: all)")
    r.add_argument("--budget", type=int, default=40, help="max judge requests this run (default 40)")
    judge_args(r)
    c = sub.add_parser("calibrate", help="judge reference parts and labelled parts, pick a threshold")
    c.add_argument("--budget", type=int, default=80)
    judge_args(c)
    for name, helptext in (("report", "pass rates from judged parts"), ("sheet", "write eval/renders/index.html")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--judge-model", help="only this judge model (default: all judge result files)")
        p.add_argument("--threshold", type=float, default=JUDGE_THRESHOLD)
    sub.choices["sheet"].add_argument("--render", default=render_tag(DEFAULT_PX),
                                      help='render setting to show (default: current), or "all"')
    args = ap.parse_args(argv)

    if args.cmd == "report":
        files = [JUDGE_DIR / f"{args.judge_model}.jsonl"] if args.judge_model else sorted(JUDGE_DIR.glob("*.jsonl"))
        print("\n".join(summarize(judged_bench(f), args.threshold) for f in files) if files else "No judge results yet.")
        return
    if args.cmd == "sheet":
        render = None if args.render == "all" else args.render
        print(f"Wrote {write_sheet(RENDERS_DIR / 'index.html', args.threshold, render).relative_to(ROOT)}")
        return

    name = judge_model(model=args.judge_model)
    judge = make_judge(model=name, prompt=args.prompt)
    judge.llm = CountingLLM(judge.llm)
    s = Session(judge, name, args.prompt, args.px, args.budget)
    (cmd_run if args.cmd == "run" else cmd_calibrate)(args, s)


if __name__ == "__main__":
    main()
