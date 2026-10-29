# Fine-tuned Qwen3.5-4B, v1 (#46)

A LoRA fine-tune of Qwen3.5-4B on 557 verified CadQuery conversations from Claude Sonnet 5.5, compared with the
untuned model and the other local models on the same 12 cases (`rag`, 3 runs each).

## Data (`finetune/data/`)

- **Requests:** 709 written by `claude-sonnet-5-5` (effort low), 20 per request, each batch steered by an area of use
  (30 listed in `finetune/make_requests.py`; 21 came up) and a category (297 medium, 217 hard, 115 simple, 80 casual).
  Requests closer than 0.5 TF-IDF cosine to a benchmark case or library example were dropped (14 dropped).
- **Builds:** each request through `generate_cad` with `claude-sonnet-5-5` as the teacher (rag, top_k 3,
  size_feedback measured), the same prompt the app sends. 593 run, 591 kept (valid, one body, at the stated size where
  there is one; 2 dropped for the wrong size). Only 17 kept conversations include a repair: Sonnet nearly always builds
  on the first try. 116 requests not run (spending cap).
- **Cost:** about $8.40 (requests $0.96, builds $7.43).
- **Split:** 557 train, 34 validation, by request id.

## Training (`finetune/train.py`)

`unsloth/Qwen3.5-4B`, 4-bit QLoRA, rank 16 on all attention and MLP projections (21M trainable parameters, 0.5%),
2 epochs, lr 2e-4 cosine, batch 1 x 8 gradient accumulation (140 steps), loss on assistant replies only, chat
template with thinking off. RTX 3080 (10 GB): **20.5 minutes**. Train loss 0.58 -> 0.38; validation loss 0.505
(epoch 1) -> 0.482 (epoch 2). Exported to GGUF Q4_K_M (2.8 GB; the vision projector did not convert, text only) and
run in Ollama as `cad-qwen3.5:4b-v1` with the stock `qwen3.5:4b` renderer, parser and sampling settings.

## Results (12 cases x 3 runs, `rag`)

| model | size | correct | built | bbox ok | first try | avg attempts | s/case | output tokens/request |
|---|---|---|---|---|---|---|---|---|
| **cad-qwen3.5:4b-v1 (this)** | 4B | **16** | 17 | 14 | 8 | 3.06 | 58 | 2,041 |
| qwen3.5:4b (untuned) | 4B | 6 | 6 | 6 | 3 | 3.67 | 51 | 1,442 |
| qwen3.5:9b | 9B | 9 | 9 | 9 | 3 | 3.56 | 84 | 1,811 |
| ornith:9b | 9B | 16 | 17 | 15 | 9 | 3.11 | 254 | 8,070 |
| gemma4:12b (app default) | 12B | 22 | 22 | 20 | 16 | 2.47 | 289 | 6,933 |
| c3d-v0:q8_0 (public fine-tune, other branch) | 4B | 8 | 10 | 5 | 7 | 3.36 | 25 | 531 |

- **6 -> 16 correct** over the untuned model, level with ornith:9b and above qwen3.5:9b, at about 5x the speed
  of ornith:9b and gemma4:12b.
- **Script errors fell from 117 to 60** failed attempts; the failures that remain are more often a part that runs but
  is not one sound solid (`invalid_mesh`, 25 vs 5), i.e. it now gets further before failing.
- Per case (out of 3): s03 3, s04 3, m05 3, m02 2, m11 2, c03 1, h02 1, m15 1; h03, h10, m03, m08 never.
  The untuned model had m03 once; the student lost it.
- gemma4:12b is still better (22) but needs 8 GB and ~5 minutes per case.

## Visual judge (gemini-3.1-flash-lite, prompt v2, renders v2, pass at 0.5)

Same 36 runs per model. "Right size and shape" = the benchmark's size check passes and the judge passes the part,
as in `rag-copying.md`. The other local models had only been judged by local judges before; they were judged here
with the same Gemini judge so the numbers compare.

| model | right size | right size and shape | right size, wrong shape |
|---|---|---|---|
| **cad-qwen3.5:4b-v1** | 16 | **8** | 8 |
| qwen3.5:4b (untuned) | 6 | 2 | 4 |
| qwen3.5:9b | 9 | 2 | 7 |
| ornith:9b | 16 | 10 | 6 |
| gemma4:12b | 22 | 16 | 6 |

- 4x the untuned model on size and shape, but half the student's right-size parts have a wrong shape (a missing or
  misplaced feature), so on shape it is a little behind ornith:9b, where it tied on size.
- Its passes are on simple and medium cases: s03 3, s04 3, m02 1, m15 1.

## Gate and general replies (`finetune/general_check.py`, `general-check.jsonl`)

10 gate conversations (complete, vague, organic, composite, picture, two-turn) and 5 general prompts, 3 runs each.

| | cad-qwen3.5:4b-v1 | qwen3.5:4b (untuned) |
|---|---|---|
| gate reply valid JSON | 30 / 30 | 30 / 30 |
| gate scope as expected | 27 / 30 | 28 / 30 |
| general prompts passed | 14 / 15 | 14 / 15 |
| CadQuery or PLAN/BBOX in non-CAD replies | 0 / 15 | 0 / 15 |
| mean seconds per gate / general reply | 27 / 49 | 32 / 31 |

- Both models call the horse on a plinth `out_of_scope` instead of `composite` twice; the student also called
  "Make me a bracket." `out_of_scope` once (within noise). Both fail one haiku (rambling into syllable counting).
- Both Python functions pass their tests in every run for both models.
- The fine-tune did not narrow the model the way C3D-v0 and the 3D-Coder were narrowed.

## Organic suite (SDF, `--suite organic --config sdf`, 12 cases x 2 runs)

Run on the code from #48 (SDF flat-base fix). The training data had no SDF code, so this checks whether the
CadQuery fine-tune helped or hurt sculpting. Scored by the same Gemini judge.

| model | built | best-effort mesh | nothing | judged right object |
|---|---|---|---|---|
| cad-qwen3.5:4b-v1 | 3 | 13 | 8 | 2 (o10 twice) |
| qwen3.5:4b (untuned) | 1 | 10 | 13 | 0 |
| gemini-3.5-flash-lite (`organic-sdf.md`, for scale) | | | | 15 of 24 built and passed |

- Neither 4B model can sculpt yet: the judge calls most parts "arbitrary geometric shapes". The fine-tune did not
  make it worse.
- The scripts import `sdf` correctly (69 of 70 attempts) but guess the API: `cylinder()` (does not exist),
  `translate(x, y, z)` instead of a tuple, `ellipsoid(a, b, c)`, multiplying a shape by a number, undefined
  variables. Round 2 adds SDF training data for this (plan in #46).

## Caveats

- **Style samples:** the request generator was shown three benchmark requests of the same category as style samples.
  No near copy was kept, but the training requests may read more like the benchmark than real users do, which could
  flatter these numbers. v2 should take the style samples from the example library instead.
- **Few repairs:** 17 of 591 conversations show a repair, so the student learned mostly to write, not to fix.
- **One run per model setting**, 36 runs each: a difference of a few cases is within run-to-run noise.
- Not yet checked: the visual judge on the student's parts, the organic (SDF) suite, and whether general replies and
  the gate's JSON still work.
