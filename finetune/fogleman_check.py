"""Check that kept SDF scripts are valid fogleman/sdf code (#46). Runs in the `finetune` environment, which has the
public fogleman/sdf installed as `sdf`; the app's environment never does.

    pixi run -e finetune ft-fogleman            # checks finetune/data/sdf/conversations.jsonl
    pixi run -e finetune ft-fogleman --examples # checks the SDF example library

Each script's last code block is run twice, once with `from sdf import *` meaning fogleman/sdf and once meaning
our stl_agent/sdf.py, and the two shapes are compared at points sampled around them: a script passes when it runs
on fogleman/sdf and the two agree on inside/outside nearly everywhere. Writes finetune/data/sdf/fogleman.jsonl
({"id", "ok", "reason", "agree", "max_diff"}); `ft-export` keeps only SDF conversations marked ok.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
SDF_DIR = HERE / "data" / "sdf"
RESULTS = SDF_DIR / "fogleman.jsonl"
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)

MIN_AGREE = 0.99  # share of sample points where both libraries say the same inside/outside


def last_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text)
    return blocks[-1] if blocks else None


def run_script(code: str, module) -> object:
    """Execute `code` with `sdf` meaning `module`; return the shape passed to show_object (or `result`)."""
    import math

    shown = []
    sys.modules["sdf"] = module
    ns = {"__name__": "__check__", "show_object": lambda s, *a, **k: shown.append(s), "math": math, "np": np}
    ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("_")})
    exec(compile(code, "<script>", "exec"), ns)
    return shown[-1] if shown else ns.get("result")


def compare(code: str, ours_mod, fog_mod, n: int = 20000, seed: int = 0) -> dict:
    try:
        ours = run_script(code, ours_mod)
    except Exception as e:
        return dict(ok=False, reason=f"fails on ours: {type(e).__name__}: {e}"[:300])
    try:
        fog = run_script(code, fog_mod)
    except Exception as e:
        return dict(ok=False, reason=f"fails on fogleman/sdf: {type(e).__name__}: {e}"[:300])
    lo, hi = ours.bounds()
    pad = 0.1 * (hi - lo) + 1
    pts = np.random.default_rng(seed).uniform(lo - pad, hi + pad, size=(n, 3))
    d_ours = np.asarray(ours(pts)).reshape(-1)
    d_fog = np.asarray(fog(pts)).reshape(-1)
    near = np.abs(d_ours) > 0.05  # points on the surface itself can fall either way
    agree = float(np.mean(np.sign(d_ours[near]) == np.sign(d_fog[near]))) if near.any() else 1.0
    max_diff = float(np.max(np.abs(d_ours - d_fog)))
    ok = agree >= MIN_AGREE
    return dict(ok=ok, reason=None if ok else f"shapes differ: inside/outside agree on {agree:.1%} of points",
                agree=round(agree, 4), max_diff=round(max_diff, 3))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true", help="check the SDF example library instead")
    ap.add_argument("--conversations", type=Path, default=SDF_DIR / "conversations.jsonl")
    args = ap.parse_args(argv)

    import sdf as fog_mod  # fogleman/sdf, installed in the finetune environment

    if not hasattr(fog_mod, "capped_cone") or "site-packages" not in getattr(fog_mod, "__file__", ""):
        raise SystemExit("`import sdf` is not fogleman/sdf: run this in the finetune environment")
    sys.path.insert(0, str(ROOT))
    from stl_agent import sdf as ours_mod

    if args.examples:
        items = [(e["id"], e["code"]) for e in json.loads((ROOT / "stl_agent/data/sdf_examples.json").read_text())]
    else:
        convs = [json.loads(l) for l in args.conversations.read_text().splitlines() if l.strip()]
        items = [(c["id"], last_code(c["messages"][-1]["content"])) for c in convs]
    done = {}
    if not args.examples and RESULTS.exists():
        done = {r["id"]: r for r in map(json.loads, RESULTS.read_text().splitlines())}
    rows = []
    for cid, code in items:
        if cid in done:
            continue
        row = dict(id=cid, **(compare(code, ours_mod, fog_mod) if code else dict(ok=False, reason="no code")))
        rows.append(row)
        print(f"{cid}: {'ok' if row['ok'] else row['reason']}" + (f" (agree {row['agree']})" if 'agree' in row else ""),
              flush=True)
    sys.modules["sdf"] = fog_mod
    if not args.examples and rows:
        with RESULTS.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)} of {len(rows)} ok")


if __name__ == "__main__":
    main()
