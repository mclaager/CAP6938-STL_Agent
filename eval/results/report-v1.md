# Report round report-v1 (#11)

One benchmark round for the final report: every model on every case, 3 runs each, on one code version with the
app's current settings. Run 2026-11-06 to 2026-11-07. Manifest: `eval/studies/report-v1.toml`; rows:
`eval/results/studies/report-v1/`. Regenerate the tables with

    pixi run -e cap6938 bench report --study report-v1 --judge-model gemini-3.8-flash --organic-judge-model claude-sonnet-5-5
    pixi run -e cap6938 judge compare --study report-v1 --judge-model gemini-3.1-flash-lite gemini-3.8-flash claude-haiku-5-5 claude-sonnet-5-5

**Status:** complete (all jobs, 2026-11-07 20:00).

## Summary

Right size and shape (main suite, judged by gemini-3.8-flash), mean over cases with 95% intervals:

| model | where it runs | right size | right size + shape |
|---|---|---|---|
| claude-sonnet-5-5 | API | 99% [96-100] | 99% [96-100] |
| gemini-3.5-flash-lite | API | 88% [81-94] | 74% [64-84] |
| gemma-4-31b-it | API (Google) | 85% [77-92] | 74% [65-83] |
| gemma4:12b (app's local default) | local, RTX 3080 | 77% [67-86] | 55% [42-67] |
| cad-qwen3.5:4b-v2 (fine-tuned, #46) | local | 51% [40-62] | 30% [19-41] |
| qwen3.5:4b (its untuned base) | local | 17% [8-26] | 13% [6-22] |

(all with 3 retrieved examples, `rag-size`)

Findings (paired comparisons by case, exact sign test; see the tables below):

- **Sonnet is near the ceiling** of this suite (99%), 24 points above Gemini Flash Lite on right size and shape
  (p < 0.001). The 48 cases no longer separate the strongest models.
- **RAG made no measurable difference for the API models:** Gemini Flash Lite 74% vs 74% (+1 pts [-8, +9]),
  Sonnet 99% vs 98%. For the small untuned local model (qwen3.5:4b) it helps it produce a valid part at all
  (12% of runs built vs 3% without examples), but the gain in correct parts is within noise: 17% vs 10%, +6 pts
  [-1, +14], p = 0.18; right shape 13% vs 10%, p = 0.51. Without examples it also spends twice the tokens per
  correct part (177K vs 85K). The model is near the floor either way, so this can't settle whether RAG helps
  small models in general.
- **Fine-tuning works:** the fine-tuned Qwen 4B triples the base model's right-size rate (51% vs 17%, +34 pts
  [+22, +47], p < 0.001) and doubles right shape (30% vs 13%, p = 0.007), at a third of the tokens per correct part.
  It is still well below the 12B local default (55% right shape).
- **Model size matters more than local vs API:** the same Gemma 4 family at 31B (on Google's servers) beats the
  local 12B on shape by 19 points [+6, +32] (p = 0.005), and matches Gemini Flash Lite.
- **Organic shapes (judged by claude-sonnet-5-5):** sculpting with Gemini Flash Lite gives a part of the right
  shape in 52% [40-64] of runs, Thingiverse's first matching download 37% [20-53] (+16 pts [-7, +38], p = 0.33:
  not significant). The small local models can't sculpt: fine-tuned Qwen 1%, base 0%.
- **Cost and speed:** Gemini Flash Lite is the cheapest correct part ($0.003) and fastest (median 3 s per build);
  Sonnet costs $0.011 per correct part at 6 s. Locally, gemma4:12b takes a median of 136 s per build (up to
  ~10 min on hard cases), mostly thinking (11.5K output tokens per run).

## Choosing the judge (#28)

The round was first judged by gemini-3.1-flash-lite, the judge of all earlier studies. Four judges then scored the
same 204 items 3 times each: 58 reference checks and 146 hand-graded parts (46 earlier, 100 new ones picked across
builders and verdicts and graded blind: no builder, config or judge score shown).

| judge | held-out accuracy | mechanical parts | organic parts | kappa vs grades | same verdict 3/3 | $/item |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 83% [77-88] | 77% | 79% | 0.54 | 95% | $0.0005 |
| gemini-3.8-flash | 92% [88-95] | 94% | 71% | 0.76 | 99% | not listed |
| claude-haiku-5-5 | 90% [85-94] | 92% | 65% | 0.70 | 96% | $0.0004 |
| claude-sonnet-5-5 | 96% [93-98] | 95% | 91% | 0.87 | 97% | $0.005 |

- The old judge was the weak link: it got 35 of the 204 items wrong; the other three fix 26-29 of them. It also
  failed Claude-built parts 50 points more often than the hand grades (12 graded Sonnet parts), so its verdicts
  would have understated Sonnet.
- No judge clearly favours its own model family (pass-rate gap to the hand grades within 0-12 points per builder
  family for the three newer judges).
- **Deviation from the rule written before the runs** ("best held-out accuracy; if its interval overlaps the old
  judge's, the cheaper one"): that rule picks Sonnet, but judging every part with it would have cost about $5.90,
  more than the remaining Anthropic budget. gemini-3.8-flash is statistically tied with Sonnet overall (92% vs 96%,
  McNemar p = 0.08) and was used for the main suite. On organic parts it is much weaker (71% vs Sonnet's 91%, 34
  graded parts), so the organic suite uses Sonnet's verdicts; the organic arms are also shown under gemini-3.8-flash.

## Changes during the round

All added after the first runs, none changes earlier rows:
- `gemma-4-31b-it` (Gemini API) main suite x 3, after a 7-run pilot: does the gap come from size or running locally?
- `qwen-base-norag`: no-RAG arm for the untuned 4B model.
- Judge choice as above; organic suite judged separately.

The build code was the same for every row (rows record their commit; later commits only changed judging and the
report output).

## Caveats

- 48 main and 30 organic cases. Intervals are over cases; repeats are not independent samples.
- One human grader (146 parts); only 34 graded organic parts, so the organic judge accuracies have wide intervals.
- Token counts compare within a model family only. gemma-4-31b-it reports few output tokens (~900 per run), so its
  thinking is likely not counted; its tokens per second are not comparable.
- Timing: local models on one RTX 3080 (Ollama, 16K context); API times depend on server load (medians reported).
  gemma-4-31b-it's long request times are Google's serving time.
- Sonnet at 99% means the main suite can't rank models above it; harder cases would be the next improvement.

## Spend

Anthropic $7.59 (Sonnet builds $3.21; judges: Sonnet $4.14, Haiku $0.22, checks $0.01). Gemini: Flash Lite builds
$1.36, gemini-3.1-flash-lite judging $0.79; gemini-3.8-flash judging and gemma-4-31b-it builds have no listed price (2.5M input /
114K output and 1.1M input / 129K output tokens). Local GPU: about 30 h.

---

# Tables

## Results per model and config

Rates are the mean over cases of each case's rate over its repeats; [brackets] are 95% bootstrap intervals over cases. Main suite: correct = built at the expected size (or declined when it should be). Organic suite: correct = built as a valid mesh (no expected size). Right shape: correct, and the visual judge (gemini-3.8-flash; organic suite: claude-sonnet-5-5) passes the part; for the organic suite, any part that came out (valid mesh or not) that the judge passes.

| model | suite | config | cases | runs | correct | right size + shape | judged | built | first try | requests/run | $/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cad-qwen3.5:4b-v2 | main | rag-size | 48 | 144 | 51% [40-62] | 30% [19-41] | 117 | 55% | 28% | 2.74 | - |
| cad-qwen3.5:4b-v2 | organic | sdf | 30 | 90 | 36% [28-43] | 1% [0-3] | 84 | 36% | 0% | 3.63 | - |
| claude-sonnet-5-5 | main | norag-size | 48 | 144 | 98% [94-100] | 98% [94-100] | 129 | 89% | 87% | 1.04 | $0.0112 |
| claude-sonnet-5-5 | main | rag-size | 48 | 144 | 99% [96-100] | 99% [96-100] | 129 | 90% | 88% | 1.02 | $0.0110 |
| gemini-3.5-flash-lite | main | norag-size | 48 | 144 | 89% [81-95] | 74% [64-83] | 124 | 81% | 42% | 1.84 | $0.0030 |
| gemini-3.5-flash-lite | main | rag-size | 48 | 144 | 88% [81-94] | 74% [64-84] | 123 | 78% | 55% | 1.62 | $0.0026 |
| gemini-3.5-flash-lite | organic | sdf | 30 | 90 | 82% [72-91] | 52% [40-64] | 90 | 82% | 1% | 2.78 | $0.0061 |
| gemma-4-31b-it | main | rag-size | 48 | 144 | 85% [77-92] | 74% [65-83] | 128 | 82% | 51% | 1.94 | - |
| gemma4:12b | main | rag-size | 48 | 144 | 77% [67-86] | 55% [42-67] | 115 | 68% | 44% | 2.03 | - |
| qwen3.5:4b | main | norag-size | 48 | 144 | 10% [4-18] | 10% [3-17] | 7 | 3% | 1% | 3.67 | - |
| qwen3.5:4b | main | rag-size | 48 | 144 | 17% [8-26] | 13% [6-22] | 28 | 12% | 4% | 3.56 | - |
| qwen3.5:4b | organic | sdf | 30 | 90 | 3% [0-8] | 0% [0-0] | 41 | 3% | 0% | 3.99 | - |
| thingiverse | organic | thingiverse-check | 30 | 30 | 67% [50-83] | 37% [20-53] | 30 | 67% | 0% | 0.00 | - |

### Organic suite under both judges

Right shape (any part that came out, passed by the judge), mean over cases [95% interval].

| model | config | claude-sonnet-5-5 | gemini-3.8-flash |
|---|---|---|---|
| cad-qwen3.5:4b-v2 | sdf | 1% [0-3] | 4% [1-9] |
| gemini-3.5-flash-lite | sdf | 52% [40-64] | 64% [51-77] |
| qwen3.5:4b | sdf | 0% [0-0] | 0% [0-0] |
| thingiverse | thingiverse-check | 37% [20-53] | 43% [27-60] |

## Tokens and cost

Per run, all attempts included; output tokens include the model's thinking. Per correct part: everything spent on the arm divided by its correct runs, so failed and repaired runs count. Each model family splits text into tokens differently, so compare token counts within a family, and cost or time across families.

| model | suite | config | input tokens/run | output tokens/run | tokens per correct part | $ per correct part |
|---|---|---|---|---|---|---|
| cad-qwen3.5:4b-v2 | main | rag-size | 5,738 | 6,961 | 25,049 | - |
| cad-qwen3.5:4b-v2 | organic | sdf | 10,687 | 4,230 | 41,953 | - |
| claude-sonnet-5-5 | main | norag-size | 1,174 | 887 | 2,104 | $0.0115 |
| claude-sonnet-5-5 | main | rag-size | 1,905 | 723 | 2,665 | $0.0112 |
| gemini-3.5-flash-lite | main | norag-size | 2,291 | 939 | 3,633 | $0.0034 |
| gemini-3.5-flash-lite | main | rag-size | 2,889 | 692 | 4,060 | $0.0029 |
| gemini-3.5-flash-lite | organic | sdf | 7,775 | 1,499 | 11,280 | $0.0074 |
| gemma-4-31b-it | main | rag-size | 7,640 | 898 | 10,078 | - |
| gemma4:12b | main | rag-size | 4,110 | 11,538 | 20,301 | - |
| qwen3.5:4b | main | norag-size | 7,586 | 10,861 | 177,097 | - |
| qwen3.5:4b | main | rag-size | 8,164 | 6,013 | 85,067 | - |
| qwen3.5:4b | organic | sdf | 13,620 | 11,120 | 742,207 | - |
| thingiverse | organic | thingiverse-check | 0 | 0 | - | - |

## Time

Seconds, median (90th percentile). Build: wall clock for one run. Request: time inside one LLM request. Waiting: rate pacing and retry back-off per run. Sandbox: running and checking scripts per run. Judge: one judge request per part. End to end: build + judge for a judged part.

| model | suite | config | build | request | waiting | sandbox | output tokens/s | judge | end to end |
|---|---|---|---|---|---|---|---|---|---|
| cad-qwen3.5:4b-v2 | main | rag-size | 43 (152) | 12 (66) | 0.0 (0.0) | 3.6 (5.0) | 116 | 5.5 (13) | 48 (135) |
| cad-qwen3.5:4b-v2 | organic | sdf | 42 (67) | 8.8 (16) | 0.0 (0.0) | 6.5 (16) | 115 | 3.0 (4.2) | 46 (70) |
| claude-sonnet-5-5 | main | norag-size | 6.4 (19) | 5.2 (17) | 0.0 (0.0) | 1.3 (1.6) | 111 | 5.1 (11) | 13 (28) |
| claude-sonnet-5-5 | main | rag-size | 5.8 (16) | 4.7 (14) | 0.0 (0.0) | 1.3 (1.4) | 106 | 5.1 (11) | 13 (27) |
| gemini-3.5-flash-lite | main | norag-size | 3.7 (12) | 1.6 (2.6) | 0.0 (0.0) | 1.4 (4.9) | 273 | 5.5 (12) | 12 (22) |
| gemini-3.5-flash-lite | main | rag-size | 3.0 (11) | 1.5 (2.1) | 0.0 (0.0) | 1.3 (4.9) | 259 | 5.5 (10.0) | 11 (19) |
| gemini-3.5-flash-lite | organic | sdf | 12 (29) | 1.9 (2.3) | 0.0 (0.0) | 6.2 (20) | 275 | 3.0 (4.7) | 15 (32) |
| gemma-4-31b-it | main | rag-size | 102 (363) | 75 (118) | 0.0 (0.0) | 2.4 (8.2) | 6 | 5.5 (10) | 112 (351) |
| gemma4:12b | main | rag-size | 136 (581) | 86 (196) | 0.0 (0.0) | 1.5 (5.2) | 46 | 5.2 (11) | 143 (550) |
| qwen3.5:4b | main | norag-size | 66 (184) | 19 (48) | 0.0 (0.0) | 4.9 (4.9) | 118 | 4.0 (5.6) | 110 (223) |
| qwen3.5:4b | main | rag-size | 94 (277) | 23 (74) | 0.0 (0.0) | 4.9 (5.0) | 51 | 5.0 (8.0) | 85 (277) |
| qwen3.5:4b | organic | sdf | 213 (377) | 53 (94) | 0.0 (0.0) | 0.2 (4.5) | 49 | 3.0 (4.2) | 241 (399) |
| thingiverse | organic | thingiverse-check | 5.0 (7.4) | - (-) | 0.0 (0.0) | 0.0 (0.0) | - | 2.9 (4.2) | 8.4 (11) |

## Comparisons (named in the manifest before the runs)

Paired by case: the mean per-case difference a - b with a 95% bootstrap interval, the cases where a or b did better, and an exact sign test on those cases (ties left out).

| comparison | metric | cases | a | b | a - b | a better / b better / tied | p |
|---|---|---|---|---|---|---|---|
| RAG vs no RAG, Gemini | correct | 48 | 88% | 89% | -1 pts [-6, +5] | 5 / 6 / 37 | 1.000 |
| RAG vs no RAG, Gemini | right size + shape | 48 | 74% | 74% | +1 pts [-8, +9] | 9 / 7 / 32 | 0.804 |
| RAG vs no RAG, Sonnet | correct | 48 | 99% | 98% | +1 pts [+0, +2] | 1 / 0 / 47 | 1.000 |
| RAG vs no RAG, Sonnet | right size + shape | 48 | 99% | 98% | +1 pts [+0, +2] | 1 / 0 / 47 | 1.000 |
| Sonnet vs Gemini | correct | 48 | 99% | 88% | +10 pts [+3, +18] | 11 / 1 / 36 | 0.006 |
| Sonnet vs Gemini | right size + shape | 48 | 99% | 74% | +24 pts [+14, +35] | 18 / 1 / 29 | 0.000 |
| gemma4:12b vs Gemini | correct | 48 | 77% | 88% | -11 pts [-19, -4] | 2 / 13 / 33 | 0.007 |
| gemma4:12b vs Gemini | right size + shape | 48 | 55% | 74% | -19 pts [-31, -8] | 5 / 19 / 24 | 0.007 |
| Gemma 4 31B (API) vs gemma4:12b (local) | correct | 48 | 85% | 77% | +8 pts [-2, +17] | 14 / 7 / 27 | 0.189 |
| Gemma 4 31B (API) vs gemma4:12b (local) | right size + shape | 48 | 74% | 55% | +19 pts [+6, +32] | 25 / 8 / 15 | 0.005 |
| RAG vs no RAG, base Qwen 4B (local) | correct | 48 | 17% | 10% | +6 pts [-1, +14] | 10 / 4 / 34 | 0.180 |
| RAG vs no RAG, base Qwen 4B (local) | right size + shape | 48 | 13% | 10% | +3 pts [-3, +10] | 6 / 3 / 39 | 0.508 |
| Fine-tuned vs base Qwen 4B | correct | 48 | 51% | 17% | +34 pts [+22, +47] | 32 / 4 / 12 | 0.000 |
| Fine-tuned vs base Qwen 4B | right size + shape | 48 | 30% | 13% | +17 pts [+6, +28] | 17 / 4 / 27 | 0.007 |
| Fine-tuned Qwen 4B vs gemma4:12b | correct | 48 | 51% | 77% | -26 pts [-38, -15] | 4 / 25 / 19 | 0.000 |
| Fine-tuned Qwen 4B vs gemma4:12b | right size + shape | 48 | 30% | 55% | -25 pts [-35, -15] | 1 / 24 / 23 | 0.000 |
| Organic: sculpted (Gemini) vs Thingiverse | correct | 30 | 82% | 67% | +16 pts [-6, +36] | 10 / 8 / 12 | 0.815 |
| Organic: sculpted (Gemini) vs Thingiverse | right size + shape | 30 | 52% | 37% | +16 pts [-7, +38] | 16 / 10 / 4 | 0.327 |
| Organic: fine-tuned vs base Qwen 4B | correct | 30 | 36% | 3% | +32 pts [+23, +41] | 24 / 1 / 5 | 0.000 |
| Organic: fine-tuned vs base Qwen 4B | right size + shape | 30 | 1% | 0% | +1 pts [+0, +3] | 1 / 0 / 29 | 1.000 |
| Organic: Gemini vs fine-tuned Qwen 4B | correct | 30 | 82% | 36% | +47 pts [+33, +59] | 25 / 3 / 2 | 0.000 |
| Organic: Gemini vs fine-tuned Qwen 4B | right size + shape | 30 | 52% | 1% | +51 pts [+40, +62] | 26 / 0 / 4 | 0.000 |

a = RAG vs no RAG, Gemini: gemini-3.5-flash-lite:rag-size vs gemini-3.5-flash-lite:norag-size; RAG vs no RAG, Sonnet: claude-sonnet-5-5:rag-size vs claude-sonnet-5-5:norag-size; Sonnet vs Gemini: claude-sonnet-5-5:rag-size vs gemini-3.5-flash-lite:rag-size; gemma4:12b vs Gemini: gemma4:12b:rag-size vs gemini-3.5-flash-lite:rag-size; Gemma 4 31B (API) vs gemma4:12b (local): gemma-4-31b-it:rag-size vs gemma4:12b:rag-size; RAG vs no RAG, base Qwen 4B (local): qwen3.5:4b:rag-size vs qwen3.5:4b:norag-size; Fine-tuned vs base Qwen 4B: cad-qwen3.5:4b-v2:rag-size vs qwen3.5:4b:rag-size; Fine-tuned Qwen 4B vs gemma4:12b: cad-qwen3.5:4b-v2:rag-size vs gemma4:12b:rag-size; Organic: sculpted (Gemini) vs Thingiverse: gemini-3.5-flash-lite:sdf vs thingiverse:thingiverse-check; Organic: fine-tuned vs base Qwen 4B: cad-qwen3.5:4b-v2:sdf vs qwen3.5:4b:sdf; Organic: Gemini vs fine-tuned Qwen 4B: gemini-3.5-flash-lite:sdf vs cad-qwen3.5:4b-v2:sdf

---

# Judge comparison

## Judges compared (204 items x 3 runs)

Items: reference solutions against their own request (should pass), against an unrelated or a similar part's request (should fail), and benchmark parts you graded (match / minor deviation pass). Held-out accuracy: each item judged with a threshold picked on the other half of the cases. [brackets]: 95% bootstrap intervals over items.

| judge | items | repeats | AUC | held-out accuracy | thresholds | kappa vs you (pass/fail) | weighted kappa vs you (bands) | same verdict every repeat | score SD | input tokens/item | output tokens/item | $/item | s/item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 1242 | 3 | 0.91 | 83% [77-88] | 0.57, 0.48 | 0.54 | 0.44 | 95% | 0.31 | 1,543 | 62 | $0.0005 | 1.8 |
| gemini-3.8-flash | 1243 | 3 | 0.98 | 92% [88-95] | 0.85, 0.59 | 0.76 | 0.74 | 99% | 0.10 | 1,543 | 69 | - | 5.4 |
| claude-haiku-5-5 | 204 | 3 | 0.97 | 90% [85-94] | 0.54, 0.61 | 0.70 | 0.65 | 96% | 0.16 | 1,148 | 494 | $0.0004 | 2.8 |
| claude-sonnet-5-5 | 415 | 3 | 0.99 | 96% [93-98] | 0.43, 0.46 | 0.87 | 0.71 | 97% | 0.14 | 1,145 | 274 | $0.0050 | 3.1 |

### Held-out accuracy by kind of item

Reference checks are hand-written mechanical parts against their own or another request; graded parts are benchmark parts you graded. [brackets]: 95% bootstrap intervals over items.

| judge | reference checks (n) | graded mechanical parts (n) | graded organic parts (n) |
|---|---|---|---|
| gemini-3.1-flash-lite | 97% [91-100] (58) | 77% [69-84] (112) | 79% [65-91] (34) |
| gemini-3.8-flash | 100% [100-100] (58) | 94% [88-98] (112) | 71% [56-85] (34) |
| claude-haiku-5-5 | 100% [100-100] (58) | 92% [87-96] (112) | 65% [47-79] (34) |
| claude-sonnet-5-5 | 100% [100-100] (58) | 95% [90-98] (112) | 91% [79-100] (34) |

### Leniency by who built the part (graded parts)

Judge pass rate minus your pass rate (held-out thresholds). Positive: the judge passes more than you do. A judge that favours its own family's parts shows a higher number there.

| judge | gemini (n) | claude (n) | local (n) |
|---|---|---|---|
| gemini-3.1-flash-lite | -3 pts (69) | -50 pts (12) | -8 pts (65) |
| gemini-3.8-flash | -4 pts (69) | +0 pts (12) | -12 pts (65) |
| claude-haiku-5-5 | +6 pts (69) | +0 pts (12) | -5 pts (65) |
| claude-sonnet-5-5 | +4 pts (69) | +0 pts (12) | +0 pts (65) |

### Pairs of judges

Right/wrong against the truth on items both judged (McNemar's exact test), and agreement of their pass/fail on every item both judged, graded or not (kappa at each judge's threshold on all items).

| judge a | judge b | items | only a right | only b right | p | kappa(a, b) |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | gemini-3.8-flash | 204 | 8 | 26 | 0.003 | 0.73 |
| gemini-3.1-flash-lite | claude-haiku-5-5 | 204 | 13 | 27 | 0.038 | 0.67 |
| gemini-3.1-flash-lite | claude-sonnet-5-5 | 204 | 3 | 29 | 0.000 | 0.68 |
| gemini-3.8-flash | claude-haiku-5-5 | 204 | 10 | 6 | 0.454 | 0.90 |
| gemini-3.8-flash | claude-sonnet-5-5 | 204 | 4 | 12 | 0.077 | 0.81 |
| claude-haiku-5-5 | claude-sonnet-5-5 | 204 | 1 | 13 | 0.002 | 0.88 |

### Items gemini-3.1-flash-lite gets wrong (35)

| item | truth | gemini-3.1-flash-lite | gemini-3.8-flash | claude-haiku-5-5 | claude-sonnet-5-5 |
|---|---|---|---|---|---|
| gemini-3.5-flash-lite/lib26-cdabac/norag/c04/rep0 | pass | wrong (0.33) | right (1.00) | right (0.89) | right (0.85) |
| gemini-3.5-flash-lite/lib26-cdabac/norag/h07/rep0 | fail | wrong (0.78) | right (0.44) | right (0.41) | right (0.26) |
| gemini-3.5-flash-lite/lib26-cdabac/norag/m13/rep0 | pass | wrong (0.15) | right (1.00) | right (0.89) | right (1.00) |
| gemini-3.5-flash-lite/lib26-cdabac/rag/c01/rep0 | fail | wrong (0.63) | right (0.56) | wrong (0.81) | wrong (0.44) |
| gemini-3.5-flash-lite/lib26-cdabac/rag/h07/rep0 | pass | wrong (0.48) | right (1.00) | right (0.89) | right (0.93) |
| gemini-3.5-flash-lite/lib26-cdabac/rag/m13/rep0 | pass | wrong (0.22) | right (1.00) | right (0.89) | right (1.00) |
| gemini-3.5-flash-lite/lib26-cdabac/rag/p01/rep0 | fail | wrong (0.52) | right (0.44) | right (0.44) | right (0.44) |
| gemini-3.5-flash-lite/lib26-cdabac/rag/p06/rep0 | fail | wrong (1.00) | wrong (0.70) | wrong (0.78) | wrong (0.63) |
| gemini-3.5-flash-lite/lib9/norag/h02/rep0 | fail | wrong (0.81) | wrong (1.00) | wrong (0.85) | wrong (0.89) |
| references/h07 | pass | wrong (0.26) | right (1.00) | right (0.89) | right (0.89) |
| references/m13 | pass | wrong (0.26) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/cad-qwen3.5:4b-v2/lib11-d5ca7f+sdfp-a4585e/sdf/o03/rep2 | pass | wrong (0.48) | wrong (0.78) | right (0.56) | right (0.48) |
| report-v1/cad-qwen3.5:4b-v2/lib11-d5ca7f+sdfp-a4585e/sdf/o10/rep1 | fail | wrong (0.56) | right (0.44) | wrong (0.67) | right (0.41) |
| report-v1/cad-qwen3.5:4b-v2/lib11-d5ca7f+sdfp-a4585e/sdf/o18/rep0 | fail | wrong (0.78) | right (0.56) | right (0.37) | right (0.37) |
| report-v1/cad-qwen3.5:4b-v2/lib11-d5ca7f+sdfp-a4585e/sdf/o19/rep1 | fail | wrong (0.78) | right (0.52) | wrong (0.59) | wrong (0.44) |
| report-v1/cad-qwen3.5:4b-v2/lib26-cdabac/rag-size/m13/rep0 | pass | wrong (0.15) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/norag-size/m13/rep0 | pass | wrong (0.30) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/norag-size/m13/rep1 | pass | wrong (0.33) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/norag-size/m13/rep2 | pass | wrong (0.26) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/rag-size/h07/rep1 | pass | wrong (0.22) | right (1.00) | right (0.85) | right (0.89) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/rag-size/m13/rep1 | pass | wrong (0.15) | right (1.00) | right (0.89) | right (1.00) |
| report-v1/claude-sonnet-5-5/lib26-cdabac/rag-size/p02/rep1 | pass | wrong (0.44) | right (1.00) | right (0.70) | right (0.78) |
| report-v1/gemini-3.5-flash-lite/lib11-d5ca7f+sdfp-a4585e/sdf/o09/rep1 | pass | wrong (0.26) | wrong (0.37) | right (0.63) | right (0.48) |
| report-v1/gemini-3.5-flash-lite/lib11-d5ca7f+sdfp-a4585e/sdf/o30/rep0 | pass | wrong (0.44) | wrong (0.41) | right (0.56) | right (0.56) |
| report-v1/gemini-3.5-flash-lite/lib26-cdabac/norag-size/c04/rep2 | pass | wrong (0.30) | right (1.00) | right (0.89) | right (0.89) |
| report-v1/gemini-3.5-flash-lite/lib26-cdabac/norag-size/h03/rep0 | pass | wrong (0.56) | wrong (0.56) | right (0.63) | right (0.74) |
| report-v1/gemini-3.5-flash-lite/lib26-cdabac/norag-size/h10/rep0 | fail | wrong (0.56) | right (0.56) | right (0.44) | right (0.44) |
| report-v1/gemini-3.5-flash-lite/lib26-cdabac/norag-size/h10/rep1 | pass | wrong (0.33) | wrong (0.41) | wrong (0.48) | wrong (0.44) |
| report-v1/gemini-3.5-flash-lite/lib26-cdabac/rag-size/h02/rep0 | fail | wrong (0.93) | wrong (0.81) | wrong (0.78) | wrong (0.85) |
| report-v1/gemma-4-31b-it/lib26-cdabac/rag-size/c04/rep1 | pass | wrong (0.22) | right (1.00) | right (0.89) | right (0.89) |
| report-v1/gemma-4-31b-it/lib26-cdabac/rag-size/h06/rep0 | pass | wrong (0.48) | right (1.00) | right (0.89) | right (0.89) |
| report-v1/gemma-4-31b-it/lib26-cdabac/rag-size/h07/rep0 | pass | wrong (0.44) | right (1.00) | right (0.89) | right (0.89) |
| report-v1/gemma4:12b/lib26-cdabac/rag-size/m13/rep0 | pass | wrong (0.33) | right (1.00) | right (0.89) | right (0.96) |
| report-v1/gemma4:12b/lib26-cdabac/rag-size/m13/rep2 | pass | wrong (0.30) | right (1.00) | right (0.89) | right (0.96) |
| report-v1/thingiverse/lib26-cdabac/thingiverse-check/o28/rep0 | pass | wrong (0.37) | wrong (0.48) | wrong (0.37) | right (0.48) |

---

# Per model and category (bench report)

## cad-qwen3.5:4b-v2 — lib11-d5ca7f+sdfp-a4585e (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| sdf | 90 | 36% | 36% | 0% | 3.63 | – | 327 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| sdf / organic | 90 | 36% | 36% | 0% | 3.63 | – | 327 |

## cad-qwen3.5:4b-v2 — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size | 144 | 51% | 55% | 28% | 2.74 | 86% | 395 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size / simple | 15 | 87% | 87% | 67% | 1.67 | 100% | 25 |
| rag-size / medium | 51 | 61% | 71% | 39% | 2.47 | 86% | 126 |
| rag-size / hard | 39 | 38% | 44% | 13% | 3.33 | 86% | 130 |
| rag-size / proposal | 24 | 29% | 17% | 4% | 3.12 | 50% | 75 |
| rag-size / casual | 15 | 47% | 60% | 33% | 2.60 | 60% | 39 |

## claude-sonnet-5-5 — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size | 144 | 98% | 89% | 87% | 1.04 | 98% | 150 |
| rag-size | 144 | 99% | 90% | 88% | 1.02 | 98% | 147 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size / simple | 15 | 100% | 100% | 100% | 1.00 | 100% | 15 |
| norag-size / medium | 51 | 100% | 100% | 98% | 1.02 | 100% | 52 |
| norag-size / hard | 39 | 95% | 100% | 100% | 1.00 | 94% | 39 |
| norag-size / proposal | 24 | 100% | 38% | 38% | 1.00 | 100% | 24 |
| norag-size / casual | 15 | 93% | 93% | 80% | 1.33 | 100% | 20 |
| rag-size / simple | 15 | 100% | 100% | 100% | 1.00 | 100% | 15 |
| rag-size / medium | 51 | 100% | 100% | 100% | 1.00 | 100% | 51 |
| rag-size / hard | 39 | 95% | 100% | 97% | 1.03 | 94% | 40 |
| rag-size / proposal | 24 | 100% | 38% | 38% | 1.00 | 100% | 24 |
| rag-size / casual | 15 | 100% | 100% | 87% | 1.13 | 100% | 17 |

## gemini-3.5-flash-lite — lib11-d5ca7f+sdfp-a4585e (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size | 144 | 89% | 81% | 42% | 1.84 | 94% | 265 |
| sdf | 90 | 82% | 82% | 1% | 2.78 | – | 250 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size / simple | 15 | 100% | 100% | 73% | 1.33 | 100% | 20 |
| norag-size / medium | 51 | 94% | 96% | 51% | 1.98 | 98% | 101 |
| norag-size / hard | 39 | 69% | 82% | 28% | 2.41 | 82% | 94 |
| norag-size / proposal | 24 | 100% | 25% | 17% | 1.12 | 100% | 27 |
| norag-size / casual | 15 | 93% | 93% | 60% | 1.53 | 100% | 23 |
| sdf / organic | 90 | 82% | 82% | 1% | 2.78 | – | 250 |

## gemini-3.5-flash-lite — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size | 144 | 89% | 81% | 42% | 1.84 | 94% | 265 |
| rag-size | 144 | 88% | 78% | 55% | 1.62 | 97% | 234 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size / simple | 15 | 100% | 100% | 73% | 1.33 | 100% | 20 |
| norag-size / medium | 51 | 94% | 96% | 51% | 1.98 | 98% | 101 |
| norag-size / hard | 39 | 69% | 82% | 28% | 2.41 | 82% | 94 |
| norag-size / proposal | 24 | 100% | 25% | 17% | 1.12 | 100% | 27 |
| norag-size / casual | 15 | 93% | 93% | 60% | 1.53 | 100% | 23 |
| rag-size / simple | 15 | 100% | 100% | 80% | 1.33 | 100% | 20 |
| rag-size / medium | 51 | 94% | 94% | 63% | 1.59 | 100% | 81 |
| rag-size / hard | 39 | 69% | 77% | 54% | 2.03 | 88% | 79 |
| rag-size / proposal | 24 | 100% | 25% | 17% | 1.25 | 100% | 30 |
| rag-size / casual | 15 | 87% | 87% | 67% | 1.60 | 100% | 24 |

## gemma-4-31b-it — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size | 144 | 85% | 82% | 51% | 1.94 | 100% | 279 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size / simple | 15 | 100% | 100% | 100% | 1.00 | 100% | 15 |
| rag-size / medium | 51 | 98% | 98% | 57% | 1.67 | 100% | 85 |
| rag-size / hard | 39 | 72% | 72% | 49% | 2.13 | 100% | 83 |
| rag-size / proposal | 24 | 62% | 46% | 21% | 2.83 | 100% | 68 |
| rag-size / casual | 15 | 93% | 93% | 40% | 1.87 | 100% | 28 |

## gemma4:12b — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size | 144 | 77% | 68% | 44% | 2.03 | 99% | 293 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| rag-size / simple | 15 | 100% | 100% | 80% | 1.27 | 100% | 19 |
| rag-size / medium | 51 | 82% | 82% | 57% | 1.84 | 100% | 94 |
| rag-size / hard | 39 | 54% | 54% | 31% | 2.79 | 100% | 109 |
| rag-size / proposal | 24 | 88% | 29% | 8% | 1.75 | 100% | 42 |
| rag-size / casual | 15 | 80% | 87% | 53% | 1.93 | 83% | 29 |

## qwen3.5:4b — lib11-d5ca7f+sdfp-a4585e (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size | 144 | 10% | 3% | 1% | 3.67 | 80% | 529 |
| sdf | 90 | 3% | 3% | 0% | 3.99 | – | 359 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size / simple | 15 | 13% | 13% | 7% | 3.67 | 100% | 55 |
| norag-size / medium | 51 | 4% | 4% | 0% | 3.92 | 100% | 200 |
| norag-size / hard | 39 | 0% | 0% | 0% | 4.00 | – | 156 |
| norag-size / proposal | 24 | 46% | 0% | 0% | 2.50 | – | 60 |
| norag-size / casual | 15 | 0% | 7% | 0% | 3.87 | 0% | 58 |
| sdf / organic | 90 | 3% | 3% | 0% | 3.99 | – | 359 |

## qwen3.5:4b — lib26-cdabac (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size | 144 | 10% | 3% | 1% | 3.67 | 80% | 529 |
| rag-size | 144 | 17% | 12% | 4% | 3.56 | 82% | 513 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| norag-size / simple | 15 | 13% | 13% | 7% | 3.67 | 100% | 55 |
| norag-size / medium | 51 | 4% | 4% | 0% | 3.92 | 100% | 200 |
| norag-size / hard | 39 | 0% | 0% | 0% | 4.00 | – | 156 |
| norag-size / proposal | 24 | 46% | 0% | 0% | 2.50 | – | 60 |
| norag-size / casual | 15 | 0% | 7% | 0% | 3.87 | 0% | 58 |
| rag-size / simple | 15 | 60% | 60% | 13% | 3.00 | 100% | 45 |
| rag-size / medium | 51 | 10% | 10% | 4% | 3.80 | 100% | 194 |
| rag-size / hard | 39 | 0% | 0% | 0% | 4.00 | – | 156 |
| rag-size / proposal | 24 | 42% | 0% | 0% | 2.75 | – | 66 |
| rag-size / casual | 15 | 0% | 20% | 13% | 3.47 | 0% | 52 |

## thingiverse — any (no-RAG rows are shared across libraries)

| config | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| thingiverse-check | 30 | 67% | 67% | 0% | 0.00 | – | 0 |

| config / category | n | correct | built | first try | avg attempts | bbox ok | requests |
|---|---|---|---|---|---|---|---|
| thingiverse-check / organic | 30 | 67% | 67% | 0% | 0.00 | – | 0 |
