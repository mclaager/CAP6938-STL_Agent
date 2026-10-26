# Visual judge on a local model (issue #19)

Can the visual judge (#8) run locally? Judge: `gemma4:12b` on Ollama (10 GB GPU, 16K context, model
reasoning on). Run on 2026-10-26 with prompt v2 and render v2 (256 px views), on the same 104 calibration items
as `judge-gemini-3.1-flash-lite.md`: 58 reference checks (22 correct parts, 22 unrelated requests, 14 near
misses) and the 46 hand-labelled parts. Command:

    pixi run -e cap6938 judge calibrate --set judge_provider=ollama --judge-model gemma4:12b --budget 120

Rows are in `eval/results/judge/gemma4:12b.jsonl`. All 104 items got a score (no errors), about 12 s each.

## Against Gemini

| | gemini-3.1-flash-lite | gemma4:12b (local) |
|---|---|---|
| AUC (how well scores separate pass from fail) | 0.96 | 0.93 |
| accuracy at 0.50 (the current `judge_threshold`) | **92%** | 88% |
| best threshold, accuracy there | 0.50, 92% | 0.94, 91% |
| hand labels right at the best threshold | 40/46 | 40/46 |
| exact band agreement with the hand grades | 67% | 72% |
| judge more lenient / harsher than the hand grade | 4 / 11 | **12** / 1 |
| bands used (match / minor / feature wrong / different) | 42 / 23 / 23 / 48 | 58 / 3 / 9 / 34 |

Overall it is close to Gemini, but it errs the other way: Gemini is a little harsh, gemma is lenient. It almost
never uses the middle bands: a part is either a full match or clearly wrong. So its best threshold is 0.94
(only a top score passes), not 0.50. To judge with it, use `--set judge_threshold=0.94`; the default stays at
0.50, which was calibrated for Gemini.

## Where it differs

**Better than Gemini on Gemini's two known errors.** The stepped shaft (m13: reference and both generated
shafts) gets 1.0 from gemma, where Gemini called it symmetrical and failed it every time. The pipe elbow
reference (h07) and the correct RAG elbow get 1.0, where Gemini failed them over the "no white background"
rule. These were the errors the Gemini write-up put down to the model's ability, not the input, and a
different model gets them right.

**Too lenient on parts with a wrong feature.** At 0.50 gemma disagrees with the hand labels on 9 parts, all
in the lenient direction; 7 of them it rates a full match (1.0):

| part | hand grade | gemma | gemini |
|---|---|---|---|
| chess rook p06, rag | feature wrong | 1.00 | 1.00 |
| phone stand h02, norag (lib9) | feature wrong | 1.00 | 0.78 |
| AA battery holder c01, rag (holes overlap in pairs) | feature wrong | 1.00 | 0.44 |
| c03, rag | feature wrong | 1.00 | 0.22 |
| c03, norag | different object | 1.00 | 0.33 |
| ruler p01, rag | feature wrong | 1.00 | 0.44 |
| pipe elbow h07, norag (ends not square) | feature wrong | 0.89 | 0.78 |
| clamp ring m08, rag (no gap) | different object | 0.56 | 0.22 |
| m11, rag (slot runs the wrong way) | different object | 0.56 | 0.22 |

On the last two it names the problem correctly ("missing the 2 mm gap", "the slot runs horizontally") but
still scores 5/10, in the "feature wrong" band, which passes at 0.50 and fails at 0.94.

**Reference checks.** At 0.94 it fails 2 of 22 correct parts: h10, where it sees one gusset instead of two,
and m17, where it reads the through-arch as a blind pocket. Gemini also marks m17 down. At 0.50 it passes
3 of 14 near misses (c02@h13, h13@m14, m14@h13). Gemini fails all three. One clear miss: the arch block
judged against the pipe elbow request (m17@h07, an unrelated object) got 10/10 with no problems listed.

## qwen3.5:9b: not usable as a judge with reasoning on

The other local vision model was stopped after 19 items. It took about 100 s per item, so the full run would
take about 3 hours. On 6 of the 7 correct reference parts it reasoned until it hit the 16K output limit (about
15,500 tokens) and never gave a score, so those count as errors. It does reject wrong parts. Its 19 rows are
in `eval/results/judge/qwen3.5:9b.jsonl`. With reasoning turned off it might work; this was not tried.

## Conclusion

The judge can run locally with `gemma4:12b` at threshold 0.94: about as accurate as Gemini on these items
(91% vs 92%) and better on the parts Gemini can't read, but more likely to pass a part with a wrong feature.
For the benchmark numbers in the reports, Gemini stays the reference judge. Two judges with opposite habits
could also be combined: a part counts as a pass only when both pass it, or a disagreement is checked by hand.

## Caveats

- One run per item; LLM output varies run to run, so a difference of one or two items is noise.
- The best threshold was picked on the same 104 items it is scored on, so 91% is optimistic. The same is true
  of Gemini's 92%.
- Hand labels are one person's grades on 46 parts.
