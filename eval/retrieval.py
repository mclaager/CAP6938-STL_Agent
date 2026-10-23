"""Offline retrieval evaluation (issue #20): no LLM requests, runs in seconds.

    python -m eval.retrieval                      # all methods
    python -m eval.retrieval --methods tfidf embed
    python -m eval.retrieval --queries casual      # everyday-language versions of the requests
    python -m eval.retrieval --queries gate        # what the app retrieves on: the gate's spec for them

Scores each retrieval method against hand-labelled relevant examples (eval/retrieval_labels.json)
for every benchmark request (eval/cases.json):

    hit@1      top-ranked example is relevant
    recall@3   share of the relevant examples that make the top 3 (same per-class cap as the app)
    MRR        mean of 1 / rank of the first relevant example
    AUC        chance a relevant example scores above an unrelated one, over all (request, example)
               pairs; independent of any cutoff
    cutoff     the similarity cutoff with the best F1 for "top 3 above the cutoff", and that F1.
               Requests with no relevant example (p03) count every retrieved example as a false positive.
    load / query   time to build the retriever (incl. loading the model) and median time per request
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from stl_agent.retrieval import RETRIEVERS

EVAL_DIR = Path(__file__).parent
LABELS_PATH = EVAL_DIR / "retrieval_labels.json"
CASES_PATH = EVAL_DIR / "cases.json"
CASUAL_PATH = EVAL_DIR / "retrieval_queries_casual.json"
GATE_PATH = EVAL_DIR / "retrieval_queries_gate.json"


METHODS = RETRIEVERS


def evaluate(make, examples: list[dict], cases: list[dict], labels: dict, k: int = 3, timing_reps: int = 20) -> dict:
    t0 = time.perf_counter()
    r = make(examples)
    load_s = time.perf_counter() - t0
    ids = [e["id"] for e in examples]

    hits, recalls, rrs, rel_scores, irr_scores, per_case = [], [], [], [], [], {}
    topk_scored = []  # (case relevant set, [(id, score) in the app's top k])
    for c in cases:
        rel = set(labels[c["id"]]["relevant"])
        s = r.scores(c["request"])
        order = [ids[i] for i in np.argsort(s)[::-1]]
        top = r.retrieve(c["request"], k=k, min_sim=-1e9)
        topk_scored.append((rel, [(e["id"], float(s[ids.index(e["id"])])) for e in top]))
        per_case[c["id"]] = [e["id"] for e in top]
        for i, eid in enumerate(ids):
            (rel_scores if eid in rel else irr_scores).append(float(s[i]))
        if not rel:
            continue
        hits.append(order[0] in rel)
        recalls.append(len(rel & {e["id"] for e in top}) / len(rel))
        rrs.append(1 / (1 + min(order.index(x) for x in rel)))

    # AUC (Mann-Whitney): share of (relevant, unrelated) score pairs ordered correctly.
    rel_a, irr_a = np.array(rel_scores), np.array(irr_scores)
    auc = float(((rel_a[:, None] > irr_a[None, :]).sum() + 0.5 * (rel_a[:, None] == irr_a[None, :]).sum())
                / (len(rel_a) * len(irr_a)))

    # Best cutoff for "keep the top k whose score is >= t".
    best = (0.0, 0.0)
    total_rel = sum(len(rel) for rel, _ in topk_scored)
    for t in sorted({sc for _, top in topk_scored for _, sc in top}):
        kept = [(eid in rel) for rel, top in topk_scored for eid, sc in top if sc >= t]
        tp = sum(kept)
        if not kept or not tp:
            continue
        p, rc = tp / len(kept), tp / total_rel
        f1 = 2 * p * rc / (p + rc)
        if f1 > best[1]:
            best = (t, f1)

    times = []
    for _ in range(timing_reps):
        for c in cases:
            t1 = time.perf_counter()
            r.retrieve(c["request"])
            times.append(time.perf_counter() - t1)

    return dict(hit1=np.mean(hits), recall3=np.mean(recalls), mrr=np.mean(rrs), auc=auc,
                cutoff=best[0], f1=best[1], load_s=load_s, query_ms=1000 * statistics.median(times),
                rel_median=float(np.median(rel_a)), irr_p90=float(np.percentile(irr_a, 90)), top=per_case)


def score_picks(cases: list[dict], labels: dict, specs: dict, fallback) -> str:
    ids = [e["id"] for e in fallback.examples]
    hits, recalls, rrs, with_picks = [], [], [], 0
    for c in cases:
        picks = specs[c["id"]].get("examples") or []
        with_picks += bool(picks)
        order = picks or [ids[i] for i in np.argsort(fallback.scores(c["request"]))[::-1]]
        rel = set(labels[c["id"]]["relevant"])
        if not rel:
            continue
        hits.append(bool(order) and order[0] in rel)
        recalls.append(len(rel & set(order[:3])) / len(rel))
        ranks = [order.index(x) for x in rel if x in order]
        rrs.append(1 / (1 + min(ranks)) if ranks else 0.0)
    return f"{np.mean(hits):.0%} | {np.mean(recalls):.0%} | {np.mean(rrs):.2f} | {with_picks}/{len(cases)}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--show-top", action="store_true", help="also list each method's top 3 per request")
    ap.add_argument("--queries", choices=["benchmark", "casual", "gate", "gate-shape"], default="benchmark",
                    help="benchmark: the CAD-worded requests in cases.json; casual: everyday-language versions; "
                         "gate: the intake gate's specs for the casual versions (eval/gate_specs.py); "
                         "gate-shape: the gate's one-line shape summaries")
    ap.add_argument("--gate-file", type=Path, default=GATE_PATH, help="gate specs file for --queries gate/gate-shape")
    args = ap.parse_args(argv)

    from stl_agent.config import EXAMPLES_PATH

    examples = json.loads(Path(EXAMPLES_PATH).read_text())
    cases = json.loads(CASES_PATH.read_text())["cases"]
    labels_doc = json.loads(LABELS_PATH.read_text())
    labels = labels_doc["labels"]
    if not labels_doc.get("reviewed"):
        print("Note: relevance labels have not been reviewed by a human yet (eval/retrieval_labels.json).\n")
    if args.queries == "casual":
        casual = json.loads(CASUAL_PATH.read_text())
        cases = [{**c, "request": casual["queries"][c["id"]]} for c in cases]
        if not casual.get("reviewed"):
            print("Note: casual queries have not been reviewed by a human yet (eval/retrieval_queries_casual.json).\n")
    specs = None
    if args.queries in ("gate", "gate-shape"):
        specs = json.loads(args.gate_file.read_text())["specs"]
        field = "spec" if args.queries == "gate" else "shape"
        missing = [c["id"] for c in cases if c["id"] not in specs]
        if missing:
            print(f"Note: no gate spec yet for {', '.join(missing)}; scoring the other {len(cases) - len(missing)} requests.\n")
        cases = [{**c, "request": specs[c["id"]].get(field) or specs[c["id"]]["spec"]} for c in cases if c["id"] in specs]

    print("| method | hit@1 | recall@3 | MRR | AUC | best cutoff (F1) | relevant median / unrelated p90 | load | per query |")
    print("|---|---|---|---|---|---|---|---|---|")
    results = {}
    for name in args.methods:
        m = results[name] = evaluate(METHODS[name], examples, cases, labels)
        print(f"| {name} | {m['hit1']:.0%} | {m['recall3']:.0%} | {m['mrr']:.2f} | {m['auc']:.2f} | "
              f"{m['cutoff']:.2f} ({m['f1']:.2f}) | {m['rel_median']:.2f} / {m['irr_p90']:.2f} | "
              f"{m['load_s']:.2f} s | {m['query_ms']:.2f} ms |")
    if specs and any("examples" in v for v in specs.values()):
        print("\nGate picks (CAD_GATE_PICKS): the examples the gate chose, in its order; when it chose none,"
              " the retriever's ranking on these queries")
        print("| method | hit@1 | recall@3 | MRR | requests with picks |\n|---|---|---|---|---|")
        for name in args.methods:
            r = METHODS[name](examples)
            print(f"| gate picks, else {name} | " + score_picks(cases, labels, specs, r) + " |")

    if args.show_top:
        for c in cases:
            rel = set(labels[c["id"]]["relevant"])
            cells = [", ".join(f"**{x}**" if x in rel else x for x in results[n]["top"][c["id"]]) for n in args.methods]
            print(f"\n{c['id']} (relevant: {', '.join(sorted(rel)) or 'none'})\n" +
                  "\n".join(f"  {n}: {cell}" for n, cell in zip(args.methods, cells)))


if __name__ == "__main__":
    main()
