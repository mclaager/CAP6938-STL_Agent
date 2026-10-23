# Retrieval evaluation (issue #20)

Library lib26-cdabac, 19 benchmark requests, `python -m eval.retrieval`. Draft labels in
`eval/retrieval_labels.json` (not yet reviewed by a human).

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
