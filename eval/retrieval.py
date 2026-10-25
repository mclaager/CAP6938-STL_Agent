"""Offline retrieval evaluation (issue #20): no LLM requests, runs in seconds.

    python -m eval.retrieval                      # all methods
    python -m eval.retrieval --methods tfidf-char
    python -m eval.retrieval --queries casual      # everyday-language versions of the requests
    python -m eval.retrieval --queries gate        # what the app retrieves on: the gate's spec for them

Scores each retrieval method against hand-labelled relevant examples (eval/retrieval_labels.json)
for every benchmark request (eval/cases.json):

    hit@1      top-ranked example is relevant
    recall@3   share of the relevant examples that make the top 3 (same per-class cap as the app)
    MRR        mean of 1 / rank of the first relevant example
    load / query   time to build the retriever and median time per request
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

    hits, recalls, rrs, per_case = [], [], [], {}
    for c in cases:
        rel = set(labels[c["id"]]["relevant"])
        s = r.scores(c["request"])
        order = [ids[i] for i in np.argsort(s)[::-1]]
        top = r.retrieve(c["request"], k=k)
        per_case[c["id"]] = [e["id"] for e in top]
        if not rel:
            continue
        hits.append(order[0] in rel)
        recalls.append(len(rel & {e["id"] for e in top}) / len(rel))
        rrs.append(1 / (1 + min(order.index(x) for x in rel)))

    times = []
    for _ in range(timing_reps):
        for c in cases:
            t1 = time.perf_counter()
            r.retrieve(c["request"])
            times.append(time.perf_counter() - t1)

    return dict(hit1=np.mean(hits), recall3=np.mean(recalls), mrr=np.mean(rrs), load_s=load_s,
                query_ms=1000 * statistics.median(times), top=per_case)


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
    ap.add_argument("--queries", choices=["benchmark", "casual", "gate"], default="benchmark",
                    help="benchmark: the CAD-worded requests in cases.json; casual: everyday-language versions; "
                         "gate: the intake gate's specs for the casual versions (eval/gate_specs.py)")
    ap.add_argument("--gate-file", type=Path, default=GATE_PATH, help="gate specs file for --queries gate")
    args = ap.parse_args(argv)

    from stl_agent.config import EXAMPLES_PATH

    examples = json.loads(Path(EXAMPLES_PATH).read_text())
    cases = json.loads(CASES_PATH.read_text())["cases"]
    labels_doc = json.loads(LABELS_PATH.read_text())
    labels = labels_doc["labels"]
    unlabelled = [c["id"] for c in cases if c["id"] not in labels]
    if unlabelled:
        print(f"Skipping {len(unlabelled)} cases without relevance labels: {', '.join(unlabelled)}\n")
        cases = [c for c in cases if c["id"] in labels]
    if not labels_doc.get("reviewed"):
        print("Note: relevance labels have not been reviewed by a human yet (eval/retrieval_labels.json).\n")
    if args.queries == "casual":
        casual = json.loads(CASUAL_PATH.read_text())
        cases = [{**c, "request": casual["queries"][c["id"]]} for c in cases]
        if not casual.get("reviewed"):
            print("Note: casual queries have not been reviewed by a human yet (eval/retrieval_queries_casual.json).\n")
    specs = None
    if args.queries == "gate":
        specs = json.loads(args.gate_file.read_text())["specs"]
        missing = [c["id"] for c in cases if c["id"] not in specs]
        if missing:
            print(f"Note: no gate spec yet for {', '.join(missing)}; scoring the other {len(cases) - len(missing)} requests.\n")
        cases = [{**c, "request": specs[c["id"]]["spec"]} for c in cases if c["id"] in specs]

    print("| method | hit@1 | recall@3 | MRR | load | per query |")
    print("|---|---|---|---|---|---|")
    results = {}
    for name in args.methods:
        m = results[name] = evaluate(METHODS[name], examples, cases, labels)
        print(f"| {name} | {m['hit1']:.0%} | {m['recall3']:.0%} | {m['mrr']:.2f} | "
              f"{m['load_s']:.2f} s | {m['query_ms']:.2f} ms |")
    if specs and any("examples" in v for v in specs.values()):
        print("\nGate picks (gate_picks setting): the examples the gate chose, in its order; when it chose none,"
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
