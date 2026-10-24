# Visual judge, first runs (issue #8)

Judge: gemini-3.1-flash-lite. Run on 2026-10-24. Renders and per-part verdicts: `eval/renders/index.html`
(`python -m eval.judge sheet`; `--render all` also shows v1).

Two render/prompt versions were run on the same parts:
- **v1:** front, right, top + three angled views (iso front, iso back, from below).
- **v2 (current):** the angled "from below" view replaced by a flat "bottom" view, and prompt v2 says that
  only flat views show a through-hole as see-through. Motivation: in the angled underside view the judge
  read through-holes as blind pockets and solid bottoms as hollow (spotted while hand-labelling).

## v1 vs v2

| | v1 | v2 |
|---|---|---|
| verdicts claiming holes aren't through / underside hollow (78 parts) | 4 | 0 |
| best calibration threshold | 0.28 | **0.50** |
| accuracy at that threshold (58 calibration items) | 97% | 97% |
| correct references passed / near misses failed | 22/22, 12/14 | 20/22, 14/14 |

- v2 fixed the honeycomb panels (0.44 → 1.00) and the pocket plate (0.33 → 1.00), which v1 failed because
  of the underside view.
- The threshold is now in a sensible place: correct parts score higher, so 0.5 separates them from near misses.
- New mistake: the pipe elbow reference is marked down because "the front view doesn't show the white
  background". A bent tube can't show background in any flat view, so the through-hole rule misleads the
  judge for bent passages.
- Remaining mistake in both versions: the stepped shaft (m13). The reference and both generated shafts are
  correct (10 mm × 20, 16 mm × 15, 8 mm × 25 from the bottom up; measured from the meshes), and the render
  shows this clearly, but the judge calls them "symmetrical with two identical ends" and fails all three.
  This is a judge capability error, not a misleading input, so it is not patched in the prompt; prompts are
  changed only when the input itself misleads (as with v1's underside view). The remedy is a stronger judge
  model, compared on the same items.
- `CAD_JUDGE_THRESHOLD` default is 0.5, from the v2 calibration. Hand labels (`eval/judge_labels.json`) are
  still to be added; rerun `calibrate` when they are.

## Benchmark parts, v2 (`judge run --gen-model gemini-3.5-flash-lite --config rag norag --reps 0`)

78 parts from the RAG ablation (one run per case; declined and failed cases have no part).

| | judge pass | judge fail |
|---|---|---|
| bbox correct | 46 | 18 |
| bbox wrong | 6 | 8 |

**18 parts with the right size fail the judge.** Checked by eye, the judge is right on, for example:
the clamp ring has no gap (rag m08), the pulley has no V-groove (rag h04), the arch block has no arch
(rag m17), the thumbwheel has no notches (norag h12), the AA battery holes overlap in pairs (rag c01),
and the headphone "hook" is a solid block (rag c05). The bounding-box check counts all of these as correct.
The judge is wrong on m13 (stepped shaft, see above).

### What this means for the RAG ablation (#7)

| over 45 buildable cases | no RAG | RAG |
|---|---|---|
| right size (the #7 measure) | 32 | **38** |
| right size and judge pass, threshold 0.5 | **24** | 22 |
| right size and judge pass, threshold 0.3 | 26 | **30** |

RAG's +12 points in `ablation-lib26.md` hold for size, but not clearly for shape: with the judge added, the
two configs are within a few parts and the leader depends on the threshold. Some of RAG's failures look like
**copying the retrieved example**: the clamp ring (top example: a washer) came out as a plain washer, the
keyhole (top example: a slot plate) as a plain slot, and the arch block (examples: plain blocks) as a plain
block. One run per case, so this is a lead, not a conclusion. Worth testing with repeats, and with a prompt
line telling the model to add every requested feature even when the examples don't have it.

## v1 results (for reference)

v1 benchmark run at its threshold (0.3): bbox correct and judge fail 7, bbox wrong and judge pass 10.
The v1 renders are in `eval/renders/v1-256px/`.

## Caveats

- One judge model, one run per case: a difference of one or two parts is noise.
- Hand labels still to come; the near-miss calibration items mostly differ in size, which the judge gets
  as text, so labels on real generated parts are the better test.
- A stronger judge (`--judge-model gemini-3.8-flash`) could be compared on the labelled set.
