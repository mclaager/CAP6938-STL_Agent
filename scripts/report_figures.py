"""Figures for the final report (#11), made from the report round's rows (study report-v1).

    pixi run -e cap6938 python -m scripts.report_figures models | judge-example | attempts | cost | organic | failures | render-fix | all

Each subcommand writes one figure's files to deliverables/report/figures/ and prints the numbers it plotted, which should match the
tables in eval/results/report-v1.md. Rates and intervals come from the same code as `bench report --study`.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from eval import stats, study  # noqa: E402
from eval.bench import part_id, read_results, results_dir  # noqa: E402
from stl_agent.config import ROOT, settings  # noqa: E402

STUDY = "report-v1"
MAIN_JUDGE = "gemini-3.8-flash"  # judges the main suite (report-v1.md, "Choosing the judge")
OUT = ROOT / "deliverables" / "report" / "figures"

# Display names, top to bottom in the model figure.
MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gemma-4-31b-it": "Gemma 4 31B",
    "gemma4:12b": "Gemma 4 12B (local)",
    "cad-qwen3.5:4b-v2": "Qwen3.5 4B fine-tuned (local)",
    "qwen3.5:4b": "Qwen3.5 4B (local)",
}

# Categorical slots 1 and 2 (checked for colour-blind separation), plus text and grid greys.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# One colour and marker per model (categorical slots 1-7, checked for colour-blind separation), the same in every
# figure that has a line per model. Lines are labelled directly: three of the colours are light on white.
MODEL_STYLE = {
    "claude-sonnet-5-5": ("#2a78d6", "o"),
    "claude-haiku-5-5": ("#eb6834", "s"),
    "gemini-3.5-flash-lite": ("#1baf7a", "^"),
    "gemma-4-31b-it": ("#eda100", "D"),
    "gemma4:12b": ("#e87ba4", "v"),
    "cad-qwen3.5:4b-v2": ("#008300", "P"),
    "qwen3.5:4b": ("#6250d6", "X"),
}

# Shorter names for labels at line ends (the caption says which run locally).
SHORT = {
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-haiku-5-5": "Haiku 5.5",
    "gemini-3.5-flash-lite": "Gemini Flash Lite",
    "gemma-4-31b-it": "Gemma 4 31B",
    "gemma4:12b": "Gemma 4 12B",
    "cad-qwen3.5:4b-v2": "Qwen 4B fine-tuned",
    "qwen3.5:4b": "Qwen 4B base",
}

COLUMN_IN = 3.25  # CVPR single-column width, inches
PAGE_IN = 6.875  # CVPR two-column (figure*) width, inches


def style():
    """Liberation Sans (bundled with the app) at caption size, fonts embedded as TrueType."""
    for f in (ROOT / "stl_agent" / "data" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.size": 0,
        "pdf.fonttype": 42,
    })


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(results_dir(STUDY).glob("*.jsonl")):
        rows += read_results(path)
    return rows


def rate(rows: list[dict], value) -> tuple[float, float, float]:
    """Mean over cases of each case's rate, with its 95% bootstrap interval (as in `bench report`)."""
    return stats.bootstrap_mean(stats.per_case(rows, value).values())


def fig_models():
    """Figure 2: right size, and right size + shape, per model (main suite, 3 retrieved examples)."""
    rows = load_rows()
    verdicts = study.judge_verdicts(read_results(results_dir(STUDY) / "judge" / f"{MAIN_JUDGE}.jsonl"),
                                 settings().judge_threshold)
    by_arm = study.arms(rows)

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.15))
    names = list(MODELS)
    offset = 0.17  # the two markers of one model sit just above and below its row
    print(f"{'model':32} {'right size':>18} {'right size + shape':>20}")
    for i, model in enumerate(names):
        rs = by_arm[(model, "main", "rag-size")]
        size = rate(rs, lambda r: r["correct"])
        shape = rate(rs, lambda r: study.shape_ok(r, verdicts))
        print(f"{model:32} {stats.fmt_rate(*size):>18} {stats.fmt_rate(*shape):>20}")
        for (mean, lo, hi), dy, color, filled in ((size, -offset, ORANGE, False), (shape, offset, BLUE, True)):
            y = i + dy
            ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot(mean * 100, y, "o", ms=4.2, mew=1.2, color=color, zorder=3,
                    mfc=color if filled else "white")

    ax.set_yticks(range(len(names)), [MODELS[m] for m in names])
    ax.set_ylim(len(names) - 0.45, -0.75)  # first model on top; room for the legend
    ax.set_xlim(0, 104)  # room for a marker at 100%
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.set_xlabel("Runs correct (mean over 48 cases, 95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Same order as within each row: right size above, right size + shape below.
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", ms=4.2, mew=1.2, mfc="white", lw=1.2, label="Right size"),
               plt.Line2D([], [], color=BLUE, marker="o", ms=4.2, mew=1.2, lw=1.2, label="Right size + shape")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.13), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0)
    save(fig, "models")


def spread(ys: list[float], gap: float) -> list[float]:
    """Label heights near ys (same order) with at least `gap` between neighbours, so line-end labels don't overlap."""
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = min(out[b], out[a] - gap)
    return out


def fig_attempts():
    """Figure: correct after k = 1..4 attempts per model; (a) main suite with 3 retrieved examples, (b) organic
    objects sculpted with SDF code (report-v1.md, "Repair loop and retries")."""
    by_arm = study.arms(load_rows())
    panels = [("(a) Main suite (right size)", "main", "rag-size"), ("(b) Organic, sculpted (valid mesh)", "organic", "sdf")]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.2), gridspec_kw={"wspace": 0.5})
    ks = [1, 2, 3, 4]
    for ax, (title, suite, config) in zip(axes, panels):
        models = [m for m in MODELS if (m, suite, config) in by_arm]
        ends = {}
        print(title)
        for model in models:
            rs = by_arm[(model, suite, config)]
            ys = [rate(rs, lambda r, k=k: bool(r["correct"]) and r["attempts"] <= k)[0] * 100 for k in ks]
            print(f"  {model:24}", " ".join(f"{y:4.0f}%" for y in ys))
            color, marker = MODEL_STYLE[model]
            ax.plot(ks, ys, color=color, lw=1.5, marker=marker, ms=4.5, mec="white", mew=0.6, zorder=3,
                    clip_on=False)
            ends[model] = ys[-1]
        for model, y in zip(ends, spread(list(ends.values()), gap=6.5)):
            ax.annotate(SHORT[model], (4, ends[model]), xytext=(4.18, y), textcoords="data", va="center",
                        fontsize=7, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2)
                        if abs(y - ends[model]) > 2 else None)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=6)
        ax.set_xlim(0.85, 4.1)
        ax.set_xticks(ks, ["1", "2", "3", "4"])
        ax.set_xlabel("Attempts allowed")
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20), [f"{y}%" for y in range(0, 101, 20)])
        ax.tick_params(axis="y", length=0, labelcolor=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Runs correct (mean over cases)")
    save(fig, "attempts")


def fig_cost():
    """Figure: right size + shape against (a) dollars per correct part, for the models with a known price, and (b)
    median build time per run, for all models (main suite, 3 retrieved examples; report-v1.md "Summary", "Tokens and
    cost", "Time")."""
    from statistics import median

    by_arm = study.arms(load_rows())
    verdicts = study.judge_verdicts(read_results(results_dir(STUDY) / "judge" / f"{MAIN_JUDGE}.jsonl"),
                                    settings().judge_threshold)
    points = {}
    print(f"{'model':24} {'right size + shape':>20} {'$/correct':>10} {'median s':>9}")
    for model in MODELS:
        rs = by_arm[(model, "main", "rag-size")]
        shape = rate(rs, lambda r: study.shape_ok(r, verdicts))
        n_ok = sum(bool(r["correct"]) for r in rs)
        usd = [r.get("usd") for r in rs]
        per_ok = sum(usd) / n_ok if n_ok and all(u is not None for u in usd) else None  # as in "Tokens and cost"
        secs = median(r["seconds"] for r in rs)
        points[model] = (shape, per_ok, secs)
        print(f"{model:24} {stats.fmt_rate(*shape):>20} {'-' if per_ok is None else f'${per_ok:.4f}':>10} {secs:9.1f}")

    # Label offsets in points (dx, dy, horizontal alignment), chosen so labels clear each other and the bars.
    place = {
        "a": {"claude-sonnet-5-5": (-7, 0, "right"), "claude-haiku-5-5": (7, 0, "left"),
              "gemini-3.5-flash-lite": (7, 0, "left")},
        "b": {"claude-sonnet-5-5": (-7, 0, "right"), "claude-haiku-5-5": (7, -3, "left"),
              "gemini-3.5-flash-lite": (7, 0, "left"), "gemma-4-31b-it": (-7, 0, "right"),
              "gemma4:12b": (-7, 0, "right"), "cad-qwen3.5:4b-v2": (7, 0, "left"), "qwen3.5:4b": (-7, 0, "right")},
    }
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.3), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.35]})
    panels = [("a", "(a) Money: API price per correct part", 1, "Dollars per correct part (log scale)",
               [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02], lambda v: f"${v:g}", (0.0005, 0.02)),
              ("b", "(b) Time: median build time per run", 2, "Seconds per build, median (log scale)",
               [2, 5, 10, 20, 50, 100, 200], lambda v: f"{v:g}", (2, 250))]
    for ax, (key, title, xi, xlabel, ticks, fmt, xlim) in zip(axes, panels):
        for model, pt in points.items():
            x = pt[xi]
            if x is None:
                continue
            (mean, lo, hi), (color, marker) = pt[0], MODEL_STYLE[model]
            ax.plot([x, x], [lo * 100, hi * 100], color=color, lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot(x, mean * 100, marker=marker, ms=5.5, color=color, mec="white", mew=0.6, ls="none", zorder=3)
            dx, dy, ha = place[key][model]
            ax.annotate(SHORT[model], (x, mean * 100), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks, [fmt(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=6)
        ax.set_ylim(0, 106)
        ax.set_yticks(range(0, 101, 20), [f"{y}%" for y in range(0, 101, 20)])
        ax.tick_params(axis="y", length=0, labelcolor=MUTED)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Right size + shape (95% interval)")
    save(fig, "cost")


# The judge example (Sec. 4, Fig. "judge"): one request, its reference solution, and a build of the right size the
# judge failed (hand grade: different object). Renders are copied as they were judged, so the figure shows exactly
# what the judge saw.
JUDGE_EXAMPLE = {
    "judge_c01_reference.png": "references/c01.png",
    "judge_c01_gemma31b.png": "report-v1/gemma-4-31b-it/lib26-cdabac/rag-size/c01-rep1.png",
}
RENDERS = ROOT / "eval" / "renders" / "v2-256px"


def fig_judge_example():
    """Copy the two c01 renders and print the judge's scores for them (all runs)."""
    import shutil

    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in JUDGE_EXAMPLE.items():
        shutil.copyfile(RENDERS / src, OUT / name)
        print(f"wrote {(OUT / name).relative_to(ROOT)}")
    parts = {"references/c01", "report-v1/gemma-4-31b-it/lib26-cdabac/rag-size/c01/rep1"}
    for r in read_results(results_dir(STUDY) / "judge" / f"{MAIN_JUDGE}.jsonl"):
        if r["part"] in parts:
            print(r["part"], f"run {r['judge_rep']}: score {r['score']} (raw {r['raw_score']:g})", r["problems"])


# Organic examples (Sec. 5, RQ5): rows chosen to match the overall result: sculpting wins (avocado), the download is
# the wrong object, its most common failure (banana), the download wins (parrot).
ORGANIC_JUDGE = "claude-sonnet-5-5"  # judges the organic suite
ORGANIC_CASES = {"o24": "Half avocado", "o22": "Banana", "o20": "Parrot"}
ORGANIC_BUILDERS = {"claude-haiku-5-5": "Haiku 5.5\n(sculpted)", "gemini-3.5-flash-lite": "Gemini Flash Lite\n(sculpted)",
                    "thingiverse": "Thingiverse\n(downloaded)"}


def iso_view(row: dict, render_png: Path):
    """The angled 'iso front' view the judge saw, without its label: rendered again from the part's STL at 512 px
    (same camera and colours), or, when the STL isn't on this machine (outputs/ is not committed), cropped from the
    judge's 256 px render with the grey label text erased (faint specks remain where it crossed the part)."""
    import numpy as np
    from PIL import Image

    from eval.judge import locate_stl
    from stl_agent.render import render_views

    stl = locate_stl(row)
    if stl is not None:
        return render_views(str(stl), px=512)["iso front (angled)"]
    print(f"  no STL for {part_id(row)}: cropping the judge's render")
    im = np.array(Image.open(render_png).convert("RGB").crop((256, 256, 512, 512))).astype(int)
    top = im[:30]
    grey = (top.max(axis=2) - top.min(axis=2) < 25) & (top.min(axis=2) < 250)  # text is grey, the part is blue
    top[grey] = 255
    return im.astype("uint8")


def fig_organic():
    """Figure: the same three organic requests sculpted by Haiku and Gemini and downloaded from Thingiverse, each in
    the angled view the judge saw, with the organic judge's score. Sculpted builders show their median-scoring run."""
    from collections import defaultdict

    rows = load_rows()
    scores, render = defaultdict(list), {}
    for r in read_results(results_dir(STUDY) / "judge" / f"{ORGANIC_JUDGE}.jsonl"):
        if r.get("kind") == "bench" and r.get("score") is not None:
            scores[r["part"]].append(r["score"])
            render[r["part"]] = ROOT / "eval" / r["render_file"]
    threshold = settings().judge_threshold
    fig, axes = plt.subplots(len(ORGANIC_CASES), len(ORGANIC_BUILDERS), figsize=(COLUMN_IN, 3.75),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.04})
    for i, case in enumerate(ORGANIC_CASES):
        for k, model in enumerate(ORGANIC_BUILDERS):
            runs = sorted(((sum(scores[pid]) / len(scores[pid]), pid, r) for r in rows
                           if r["model"] == model and r["case"] == case and (pid := part_id(r)) in scores),
                          key=lambda t: t[:2])
            score, pid, row = runs[(len(runs) - 1) // 2]  # median run (lower middle when even)
            print(f"{case} {model:22} {score:.2f}  {pid}")
            if row.get("stl") and (src := (ROOT / row["stl"]).with_name("source.json")).exists():
                info = json.loads(src.read_text())  # downloads: credit them in the caption
                print(f"    {info['name']!r} by {info['author']}, {info['license']}, {info['url']}")
            ax = axes[i][k]
            ax.imshow(iso_view(row, render[pid]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            verdict = "pass" if score >= threshold else "fail"
            ax.set_xlabel(f"{score:.2f} ({verdict})", fontsize=7, color=INK, labelpad=2)
            if i == 0:
                ax.set_title(ORGANIC_BUILDERS[model], fontsize=7, color=INK, pad=4)
            if k == 0:
                ax.set_ylabel(ORGANIC_CASES[case], fontsize=7.5, color=INK)
    save(fig, "organic")


# Failure examples (Sec. 6): right size, wrong shape, the commonest failure of the mid-strength and strong models.
# Each pattern recurs across runs: Gemini left out h12's features in all 3 runs, Haiku made h08's holes blind in 2
# and built c03 as a plain wedge in all 3. (label, part id, views)
FAILURES = [
    ("(a) Features left out", "report-v1/gemini-3.5-flash-lite/lib26-cdabac/rag-size/h12/rep0",
     ["iso front (angled)"]),
    ("(b) Holes stop short: angled / bottom", "report-v1/claude-haiku-5-5/lib26-cdabac/rag-size/h08/rep1",
     ["iso front (angled)", "bottom"]),
    ("(c) Casual request, built literally", "report-v1/claude-haiku-5-5/lib26-cdabac/rag-size/c03/rep0",
     ["iso front (angled)"]),
]


def print_failure_counts(rows: list[dict], verdicts: dict):
    """How each main-suite run (rag-size, cases that should be built) ended, per model (report-v1.md, "How the
    main-suite runs failed")."""
    from collections import Counter

    print("model | right size + shape | right size, wrong shape | wrong size | no part: last failure")
    for model in MODELS:
        ends = Counter()
        for r in study.arms(rows)[(model, "main", "rag-size")]:
            if r.get("expect") != "build":
                continue
            ok = study.shape_ok(r, verdicts)
            if ok:
                ends["right size + shape"] += 1
            elif r["outcome"] != "built":
                ends["no part: " + (r.get("failures") or ["?"])[-1]] += 1
            elif not r["correct"]:
                ends["wrong size"] += 1
            elif ok is False:
                ends["right size, wrong shape"] += 1
            else:
                ends["not judged"] += 1
        print(f"  {model}: {dict(ends)}")


def fig_failures():
    """Figure: three parts built at the right size that the judge failed, rendered from their STLs (views as the
    judge saw them, without labels)."""
    from collections import defaultdict

    from eval.judge import locate_stl
    from stl_agent.render import render_views

    rows = load_rows()
    by_id = {part_id(r): r for r in rows}
    judge_rows = read_results(results_dir(STUDY) / "judge" / f"{MAIN_JUDGE}.jsonl")
    judged = defaultdict(list)
    for r in judge_rows:
        if r.get("kind") == "bench" and r.get("score") is not None:
            judged[r["part"]].append(r)
    print_failure_counts(rows, study.judge_verdicts(judge_rows, settings().judge_threshold))
    n_images = sum(len(views) for _, _, views in FAILURES)
    fig, axes = plt.subplots(1, n_images, figsize=(PAGE_IN, 1.75), gridspec_kw={"wspace": 0.06})
    i = 0
    for label, pid, views in FAILURES:
        row, js = by_id[pid], judged[pid]
        print(f"{pid}: correct={row['correct']}, judge {sum(j['score'] for j in js) / len(js):.2f}, {js[0]['problems']}")
        images = render_views(str(locate_stl(row)), px=512)
        for n, view in enumerate(views):
            ax = axes[i]
            ax.imshow(images[view])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if n == 0:
                ax.set_title(label, loc="left", fontsize=7.5, color=INK, pad=4)
            i += 1
    save(fig, "failures")


# Render v1 vs v2 (Sec. 4): the reference solution for h08 (real through-holes) under the first judge,
# gemini-3.1-flash-lite, during development (eval/results/judge-gemini-3.1-flash-lite.md). v1's angled view from below
# made it call the holes blind pockets; v2's flat bottom view shows the background through them.
RENDER_FIX = [
    ("v1: angled, from below", "v1-256px", (512, 256, 768, 512)),  # bottom-right cell of the v1 sheet
    ("v2: flat, bottom", "v2-256px", (0, 256, 256, 512)),  # bottom-left cell of the v2 sheet
]


def fig_render_fix():
    """Figure: the same correct part's underside in render v1 and v2, with the first judge's score for each."""
    import numpy as np
    from PIL import Image

    judged = {}
    for r in read_results(ROOT / "eval" / "results" / "judge" / "gemini-3.1-flash-lite.jsonl"):
        if r["part"] == "references/h08" and r.get("score") is not None:
            judged[r["render"]] = r
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.75), gridspec_kw={"wspace": 0.06})
    for ax, (title, render, box) in zip(axes, RENDER_FIX):
        r = judged[render]
        print(f"{render} (prompt {r['prompt']}): score {r['score']}  {r['problems']}")
        im = np.array(Image.open(ROOT / "eval" / r["render_file"]).convert("RGB").crop(box))[30:]  # drop the label
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=4)
        ax.set_xlabel(f"judge: {r['score']:.2f} ({'pass' if r['score'] >= 0.5 else 'fail'})", fontsize=7, color=INK,
                      labelpad=2)
    save(fig, "render_fix")


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)  # preview only
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def tex(text: str) -> str:
    """Request text for LaTeX: straight quotes become `` and ''."""
    out, opening = [], True
    for ch in text:
        if ch == '"':
            out.append("``" if opening else "''")
            opening = not opening
        else:
            out.append(ch)
    return "".join(out).replace("%", r"\%").replace("&", r"\&")


def fig_appendix():
    """Appendix files in deliverables/report/appendix/: every benchmark case (cases.tex) and the three prompts as plain text.
    The report merges the data's `proposal` category into `casual` (both are loosely worded requests)."""
    from stl_agent import generate
    from stl_agent.judge import DEFAULT_PROMPT
    out = ROOT / "deliverables" / "report" / "appendix"
    out.mkdir(exist_ok=True)
    main = json.loads((ROOT / "eval" / "cases.json").read_text())["cases"]
    organic = json.loads((ROOT / "eval" / "organic_cases.json").read_text())["cases"]
    lines = [f"% Made by `python -m scripts.report_figures appendix` from eval/cases.json and eval/organic_cases.json.",
             r"\subsection*{Main suite (48 cases)}", r"{\small"]
    for c in main:
        category = "casual" if c["category"] == "proposal" else c["category"]
        size = (r"\,$\times$\,".join(f"{v:g}" for v in c["expected_bbox_mm"]) + r"\,mm"
                if c.get("expected_bbox_mm") else "no fixed size")
        label = f"{category}, decline" if c["expect"] == "decline" else f"{category}, {c['expect']}, {size}"
        lines.append(rf"\noindent\textbf{{{c['id']}}} \emph{{{label}}}: {tex(c['request'])}\par")
    lines += ["}", r"\subsection*{Organic suite (30 cases)}", r"{\small"]
    for c in organic:
        lines.append(rf"\noindent\textbf{{{c['id']}}} \emph{{search: {c['search']}}}: {tex(c['request'])}\par")
    lines.append("}")
    (out / "cases.tex").write_text("\n".join(lines) + "\n")
    (out / "prompt_build.txt").write_text(generate.SYSTEM_PROMPT.strip() + "\n")
    (out / "prompt_sdf.txt").write_text(generate.SYSTEM_PROMPT_SDF.strip() + "\n")
    judge = ROOT / "stl_agent" / "judge" / "prompts" / f"{DEFAULT_PROMPT}.txt"
    (out / "prompt_judge.txt").write_text(judge.read_text().strip() + "\n")
    print(f"wrote {out}: {len(main)} main and {len(organic)} organic cases, prompts (judge {DEFAULT_PROMPT})")


FIGURES = {"models": fig_models, "judge-example": fig_judge_example, "attempts": fig_attempts, "cost": fig_cost, "organic": fig_organic, "failures": fig_failures, "render-fix": fig_render_fix,
           "appendix": fig_appendix}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("figure", choices=[*FIGURES, "all"])
    args = p.parse_args(argv)
    style()
    for name, make in FIGURES.items():
        if args.figure in (name, "all"):
            make()


if __name__ == "__main__":
    main()
