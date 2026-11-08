"""Statistics for benchmark and judge results, with the case as the sampling unit.

Repeated runs of one case are not independent samples: a model that fails a case tends to fail it every time.
So a success rate here is the mean over cases of each case's rate over its repeats, and confidence intervals
come from a bootstrap that resamples cases (repeats stay together). Paired comparisons use the cases both arms
ran: the mean per-case difference with a bootstrap CI, and an exact sign test on the cases where they differ.

numpy only (no scipy), so the app's environment can run it.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np

N_BOOT = 10_000
SEED = 0


def per_case(rows: list[dict], value) -> dict[str, float]:
    """Each case's mean of `value(row)` over its rows (repeats). Rows where value returns None are left out."""
    by_case = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            by_case[r["case"]].append(float(v))
    return {c: sum(vs) / len(vs) for c, vs in by_case.items()}


def bootstrap_mean(values, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high): the mean of `values` (one per case) and a percentile bootstrap CI over cases."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(x.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def sign_test(better: int, worse: int) -> float:
    """Exact two-sided sign test p-value: how likely a split at least this uneven is if neither arm is better.
    Ties are left out before calling. Also McNemar's exact test, with the two discordant counts."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: dict[str, float], b: dict[str, float], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Compare two arms on the cases both have (per-case rates). Returns the shared case count, each arm's mean
    on those cases, the mean difference a - b with its bootstrap CI, cases where a / b is better / tied, and the
    sign-test p-value."""
    cases = sorted(set(a) & set(b))
    d = [a[c] - b[c] for c in cases]
    mean, lo, hi = bootstrap_mean(d, n_boot, seed)
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return dict(n=len(cases), a=float(np.mean([a[c] for c in cases])) if cases else float("nan"),
                b=float(np.mean([b[c] for c in cases])) if cases else float("nan"),
                diff=mean, lo=lo, hi=hi, better=better, worse=worse, tied=len(cases) - better - worse,
                p=sign_test(better, worse))


def cohen_kappa(a: list, b: list, categories: list | None = None, weights: str | None = None) -> float:
    """Agreement between two raters beyond chance: 1 = perfect, 0 = chance level. `weights` "linear" for ordered
    categories (then a near miss counts as partly agreeing); `categories` gives their order."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equally long, non-empty lists")
    cats = categories or sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    w = np.abs(i - j) / (k - 1) if weights == "linear" and k > 1 else (i != j).astype(float)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired yes/no results of two raters (or systems) on the same items: the discordant counts and McNemar's
    exact p-value (sign test on the items where they differ)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return dict(n=len(a), only_a=only_a, only_b=only_b, p=sign_test(only_a, only_b))


def quantiles(values) -> tuple[float, float] | tuple[None, None]:
    """(median, 90th percentile), or (None, None) without values."""
    x = [float(v) for v in values if v is not None]
    if not x:
        return None, None
    return float(np.median(x)), float(np.percentile(x, 90))


def fmt_rate(mean: float, lo: float, hi: float) -> str:
    """"86% [80-91]" """
    if mean != mean:  # nan
        return "-"
    return f"{mean:.0%} [{lo * 100:.0f}-{hi * 100:.0f}]"


def fmt_diff(p: dict) -> str:
    """"+12 pts [3, 21]" """
    if p["n"] == 0:
        return "-"
    return f"{p['diff'] * 100:+.0f} pts [{p['lo'] * 100:+.0f}, {p['hi'] * 100:+.0f}]"


def fmt_s(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}" if x >= 10 else f"{x:.1f}"
