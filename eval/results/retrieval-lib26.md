# Retrieval evaluation (issue #20)

Library lib26-cdabac, 19 benchmark requests, `python -m eval.retrieval`. Labels in
`eval/retrieval_labels.json` (reviewed).

| method | hit@1 | recall@3 | MRR | AUC | best cutoff (F1) | relevant median / unrelated p90 | load | per query |
|---|---|---|---|---|---|---|---|---|
| tfidf | 72% | 60% | 0.82 | 0.88 | 0.17 (0.59) | 0.17 / 0.11 | 0.72 s | 0.46 ms |
| tfidf-char | 83% | 73% | 0.90 | 0.90 | 0.25 (0.67) | 0.28 / 0.16 | 0.00 s | 0.52 ms |
| embed | 89% | 73% | 0.94 | 0.84 | 0.76 (0.56) | 0.81 / 0.80 | 0.52 s | 5.37 ms |
| hybrid | 89% | 76% | 0.92 | 0.91 | 0.53 (0.67) | 0.54 / 0.47 | 0.27 s | 5.82 ms |

## Embedding model sweep (same labels, 2026-10-23)

Six small fastembed models x three text variants. "no numbers" strips sizes before embedding
(now the default). Picking the best of 18 variants on the same 18 labelled requests flatters
the winner; one request is ~6 points of hit@1.

| model | text | hit@1 | recall@3 | MRR | AUC | per query |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 (67 MB) | req+kw | 67% | 59% | 0.78 | 0.86 | 6.5 ms |
| bge-small-en-v1.5 | no numbers | 78% | 64% | 0.83 | 0.88 | 5.2 ms |
| all-MiniLM-L6-v2 (90 MB) | no numbers | 67% | 67% | 0.80 | 0.85 | 2.3 ms |
| **snowflake-arctic-embed-s (130 MB)** | **no numbers** | **89%** | **73%** | **0.94** | 0.84 | 4.5 ms |
| bge-base-en-v1.5 (210 MB) | req+kw | 89% | 69% | 0.93 | 0.85 | 15.6 ms |
| bge-base-en-v1.5 | no numbers | 83% | 79% | 0.90 | 0.89 | 12.8 ms |
| nomic-embed-text-v1.5-Q (130 MB) | req+kw | 83% | 59% | 0.86 | 0.80 | 12.1 ms |
| granite-embedding-small-english-r2 (180 MB) | no numbers | 89% | 69% | 0.92 | 0.83 | 6.6 ms |

Embedding request text without the keywords was worse for every model (MRR 0.64-0.70).

## Everyday-language queries (`--queries casual`, reviewed)

Same parts and labels, phrased by purpose without CAD terms (eval/retrieval_queries_casual.json).

| method | hit@1 | recall@3 | MRR | AUC | best cutoff (F1) | relevant median / unrelated p90 | load | per query |
|---|---|---|---|---|---|---|---|---|
| tfidf | 22% | 44% | 0.48 | 0.68 | 0.00 (0.30) | 0.06 / 0.08 | 0.70 s | 0.42 ms |
| tfidf-char | 56% | 44% | 0.70 | 0.75 | 0.13 (0.37) | 0.13 / 0.13 | 0.00 s | 0.48 ms |
| embed | 33% | 37% | 0.52 | 0.72 | 0.76 (0.34) | 0.77 / 0.79 | 0.46 s | 3.88 ms |
| hybrid | 50% | 39% | 0.66 | 0.78 | 0.43 (0.31) | 0.44 / 0.45 | 0.24 s | 4.69 ms |

Every method drops sharply. Embeddings help where the everyday meaning is close to the shape
("traffic cone" -> funnel, "ruler that says ALEX" -> text) but fail where a purpose must become a
shape ("hang my headphones" -> hook, "prop my phone up" -> angled stand). Some letter-chunk hits
are coincidences ("prop" ~ "profile"). In the app, retrieval runs on the intake gate's spec, which
restates the request geometrically, so the realistic test is casual request -> gate spec -> retrieval.

## Gate specs for the everyday queries (`--queries gate`)

What the app actually retrieves on: each casual query sent through the intake gate once
(gemini-3.6-flash, `python -m eval.gate_specs`, saved in eval/retrieval_queries_gate.json).

| method | hit@1 | recall@3 | MRR | AUC | best cutoff (F1) | relevant median / unrelated p90 | load | per query |
|---|---|---|---|---|---|---|---|---|
| tfidf | 44% | 63% | 0.65 | 0.88 | 0.11 (0.52) | 0.13 / 0.10 | 0.70 s | 0.46 ms |
| tfidf-char | 56% | 58% | 0.71 | 0.83 | 0.17 (0.45) | 0.25 / 0.22 | 0.00 s | 0.67 ms |
| embed | 44% | 41% | 0.57 | 0.77 | 0.80 (0.35) | 0.77 / 0.78 | 0.50 s | 10.80 ms |
| hybrid | 56% | 66% | 0.72 | 0.85 | 0.46 (0.51) | 0.50 / 0.48 | 0.27 s | 10.18 ms |

The gate's restatement recovers much of what casual wording lost (word TF-IDF recall@3 50% -> 63%,
AUC 0.68 -> 0.88), but top-match accuracy stays well below the CAD-worded requests. Two reasons:
specs are long (median 88 words) and full of extra detail that dilutes matching, embeddings most of
all; and the gate sometimes designs a different part than the labels assume (the spacer became a
wall bracket with countersunk holes), so some "misses" are label mismatches. No method clearly wins:
differences are 1-2 of 18 queries. A short shape summary from the gate (e.g. "curved hook arm on a
wall plate") would likely retrieve better than the full spec, at no extra request.
