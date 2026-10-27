# External RAG pool: Text-to-CadQuery (2026-10-27)

Question: does adding a public text→CadQuery dataset to the RAG library help? The built-in library has 26
hand-checked examples (lib26-cdabac).

**Answer: no measurable gain, so the code was removed** (importer `scripts/import_external.py`, setting
`rag_library`, bench config `rag-ext`, never committed). On the cases where a pool example is
retrieved, the two configs tie (17/21 correct each). Overall success is within run-to-run noise.

## Pool
Source: Text-to-CadQuery (arXiv 2505.06507, `huggingface.co/ricemonster/IEEE-T-ASE`), 170K CadQuery scripts
written by Gemini 2.0 Flash for Text2CAD descriptions (CC BY-NC-SA 4.0, so the data is not committed). The pool
was built by the (removed) importer, pool `t2cq96-7478c4`; `rag-ext` rows record it as `pool`.

| step | scripts |
|---|---|
| train split | 99,236 |
| short, few `lineTo`, a feature beyond plain extrude, only cadquery/math imports | 48,393 |
| Text2CAD states one overall size | 10,328 |
| one script per abstract description, shuffled, first 4000 run | 3,962 |
| **build one watertight body of the stated size (±5%)** | **96 (2.4%)** |

Most scripts run but build the wrong size. In a sample of 80, 78 ran, but only 0 of those 78 matched the size
in the Text2CAD text. The error is systematic: the sketch scale is applied twice (0.75 long → 0.5625), or small
numbers are squared (0.106 → 0.011). The extrude height is usually right. Examples like these would teach the
model wrong arithmetic, so the size check stays.

The 96 survivors are mostly extruded 2D outlines: 72 use `threePointArc`, 25 `circle`, 8 `hole`, 6
`pushPoints`, 4 `fillet`. Sizes are in dataset units (~0.75), not mm.

## Offline retrieval (`retrieval-eval` with the pool added, no requests)
External examples are never labelled relevant, so these scores only show how much the pool pushes out
built-in examples. They don't show whether it helps.

| queries | library | tfidf hit@1 | tfidf recall@3 | external in top 3 |
|---|---|---|---|---|
| benchmark wording | builtin | 72% | 60% | 0% |
| benchmark wording | builtin+external | 67% | 63% | 7% |
| casual wording | builtin | 22% | 50% | 0% |
| casual wording | builtin+external | 22% | 30% | 35% |

Adding the pool changed the top 3 for 27 of 48 benchmark cases, but only 7 got a pool example: s03, m16,
h03, p05, p08, c03, c05. For the other 20, the larger corpus changed the TF-IDF word weights and reordered the
built-in examples.

## Benchmark (gemini-3.5-flash-lite, `bench run --config rag-ext`)
All 48 cases, one run each (76 requests):

| config | runs | correct | first try | mean attempts |
|---|---|---|---|---|
| rag, rep 0 | 48 | 85% | 50% | 1.77 |
| rag, all reps | 82 | 90% | 60% | 1.59 |
| rag-ext | 48 | 90% | 58% | 1.58 |

Six cases differ between `rag` rep 0 and `rag-ext`. h07 is one of them although its retrieved examples are
the same in both, so single-run differences are noise.

The 7 cases with a pool example, 3 runs each:

| case | rag correct | rag-ext correct | rag attempts | rag-ext attempts |
|---|---|---|---|---|
| s03 hex prism | 3/3 | 3/3 | 1.0 | 1.0 |
| m16 star | 2/3 | 1/3 | 3.7 | 4.0 |
| h03 wall hook | 1/3 | 2/3 | 4.0 | 2.3 |
| p05 chess knight (decline) | 3/3 | 3/3 | 1.0 | 1.0 |
| p08 fox (decline) | 3/3 | 3/3 | 1.0 | 1.0 |
| c03 phone prop | 3/3 | 3/3 | 1.7 | 2.3 |
| c05 desk hook | 2/3 | 2/3 | 4.0 | 3.7 |
| **total** | **17/21** | **17/21** | 2.3 | 2.2 |

## Why it doesn't help
- **Wording:** our requests name a part and give its size in mm. Text2CAD describes geometry ("a rectangular
  prism with a curved bottom edge") in units around 0.75. Few pool examples match a real request, and with
  everyday wording the ones retrieved are mostly unrelated (35% of slots, recall 50% → 30%).
- **Content:** the code that builds the right size is mostly sketch + extrude. That adds little to a library
  that already covers holes, patterns, shells, revolves, text and gears.
- **Quality:** 97.6% of the dataset's scripts don't build their own stated size.

A larger pool (more sandbox runs) would add more extruded outlines of the same kind. A dataset written for
functional parts, or more hand-written examples for the cases that fail (#29), is more likely to help.
