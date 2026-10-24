# Visual judge, first run (issue #8)

Judge: gemini-3.1-flash-lite, prompt v1, render v1 (six views, 256 px each). Run on 2026-10-24.
Renders and per-part verdicts: `eval/renders/index.html` (`python -m eval.judge sheet`).

## Calibration (`python -m eval.judge calibrate`)

58 items from the 22 reference solutions: each judged against its own request (should pass), an
unrelated case's request (should fail), and for 14 of them a similar part's request (near miss, should fail).

| threshold | accuracy | references passed | unrelated failed | near misses failed |
|---|---|---|---|---|
| 0.17 | 95% | 22/22 | 22/22 | 11/14 |
| **0.28** | **97%** | 22/22 | 22/22 | 12/14 |
| 0.50 | 95% | 19/22 | 22/22 | 14/14 |
| 0.61 | 88% | 15/22 | 22/22 | 14/14 |

AUC 1.00. Default `CAD_JUDGE_THRESHOLD` is 0.3 (provisional, see caveats).

- The threshold is low because this judge often marks correct parts down with confident wrong claims:
  the honeycomb holes "aren't through-holes", the keyhole slot "points the wrong way", the 60 mm arch is
  "impossible" in a 40 mm block. 7 of 22 correct references scored 0.56 or less.
- Near misses are mostly caught through the measured size given in the prompt ("70 x 50 instead of
  60 x 40"), not the shape. Generated parts usually have the right size, so hand labels on real
  generated parts are the better test of the threshold.

## Benchmark parts (`python -m eval.judge run --gen-model gemini-3.5-flash-lite --config rag norag --reps 0`)

78 parts from the RAG ablation (one run per case; declined and failed cases have no part).

| config | judged | judge pass | bbox correct | bbox correct + judge pass |
|---|---|---|---|---|
| norag | 39 | 85% | 74% | 69% |
| rag | 39 | 87% | 90% | 77% |

| | judge pass | judge fail |
|---|---|---|
| bbox correct | 57 | 7 |
| bbox wrong | 10 | 4 |

- **Right size, wrong shape (7):** the case the bounding-box check can't see. Checked by eye:
  the thumbwheel has no rim notches (norag h12), the AA battery holes overlap in pairs (rag c01), and the
  headphone "hook" is a solid block with no hook (rag c05); the judge is right on all three. It is wrong
  on rag m13 (calls a correct stepped shaft symmetrical; same mistake as on the reference) and rag m17
  (says the arch is missing). The two phone stands (c03) are judgement calls.
- **Bbox wrong, judge pass (10):**
  - 6 are clearly wrong parts that scored only 0.33-0.44, just above the 0.3 threshold: brackets 95 mm and
    60 mm tall instead of 50 / 40 (h10), a collar twice as long (m12), an oversized handle (h09), a
    misshapen elbow (h07), a short spring (h05). On generated parts, the provisional threshold is too lenient.
  - m16 (star) is right: correct size and shape, but the pipeline rejected it against the model's own
    wrong size claim, so the bench counts it as not correct (issue #23).
  - m14 is 3 mm too tall (28 vs 25), and the two phone stands (h02) have no expected size.
- With RAG, more parts are both the right size and the right shape (77% vs 69%), in line with the
  ablation (`ablation-lib26.md`).

## Caveats and next steps

- **Threshold is provisional** until hand labels are added: label ~20 generated parts on the review page,
  paste the JSON into `eval/judge_labels.json`, and rerun `calibrate`.
- One judge model. The false complaints suggest trying a stronger judge on the same parts
  (`--judge-model gemini-3.8-flash`; its ~20/day quota covers the 20-part labelled set).
- One run per case; a one-part difference between configs is noise.
