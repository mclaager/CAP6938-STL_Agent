# Fine-tuned students, round 2 (#46)

Round 2 adds organic (SDF) training data and more CadQuery data to round 1 (`finetune-v1.md`), and trains two
bases on the RTX 3080: Qwen3.5-4B (round 1's base) and Granite 4.1 8B. Gemma 4 12B and E4B, Qwen3.5-4B in 16-bit
and the 9B models do not fit in its 9.7 GB for training (`9f06736`); Gemma 4 12B needs a rented 24 GB GPU.

## Data (`finetune/data/`, `finetune/data/sdf/`)

| | requests | teacher runs | kept | notes |
|---|---|---|---|---|
| CadQuery, round 1 | 709 (Sonnet) | 593 (Sonnet) | 591 | style samples were benchmark requests |
| CadQuery, new | 300 (Gemini) | 300 (gemini-3.5-flash-lite) | 231 | style samples are library examples |
| SDF (organic) | 507 (Gemini) | 400 (Gemini) + 65 (Sonnet, on Gemini's failures) | 178 + 53 | kept only if the Gemini judge passes the part |

- **SDF scripts stay public:** the sandbox stands every shape on z = 0 (PR #51), so scripts call nothing that
  only our `sdf.py` has; every kept script was also run on the real fogleman/sdf (`finetune/fogleman_check.py`):
  229 of 230 give the same shape (one blended script differs on 1.6% of sample points and is left out).
- **Organic requests** never name an organic benchmark subject (duck, cat, horse, ...), and are dropped when too
  close to a benchmark case or an example.
- **Repairs:** 175 of the 178 Gemini SDF conversations and 57 of the 231 new CadQuery ones include a repair (round 1:
  17 of 591).
- **Cost:** Gemini (paid tier from day 2) ~$3; Sonnet $4.45 (fallback capped at $5).
- **Training set:** 990 train / 62 validation conversations (822 CadQuery, 230 SDF), up to 6,181 tokens.

## Training (`finetune/train.py`, 4-bit QLoRA, rank 16, 2 epochs, lr 2e-4, batch 1 x 8)

| student | train | peak GPU | train loss | validation loss (epoch 1 / 2) |
|---|---|---|---|---|
| cad-qwen3.5:4b-v2 | 42.6 min | 6.1 GB | 0.430 | 0.490 / 0.474 |
| cad-granite4.1:8b-v2 | 78.4 min | 7.5 GB | 0.515 | 0.589 / 0.568 |

GGUF Q4_K_M in Ollama with each base's stock template and sampling settings. Losses are not comparable across
bases (different tokenizers).

## Main suite (12 cases x 3 runs, `rag`; judge gemini-3.1-flash-lite, prompt v2, pass at 0.5)

| model | right size | right size and shape | first try | attempts | s/case | output tokens/request |
|---|---|---|---|---|---|---|
| **cad-qwen3.5:4b-v2** | **21** | 9 | 8 | 2.94 | 65 | 2,449 |
| cad-qwen3.5:4b-v1 | 16 | 8 | 8 | 3.06 | 58 | 2,041 |
| qwen3.5:4b (untuned) | 6 | 2 | 3 | 3.67 | 51 | 1,442 |
| **cad-granite4.1:8b-v2** | 19 | **10** | 11 | 2.75 | **32** | 452 |
| granite4.1:8b (untuned) | 11 | 2 | 7 | 3.33 | 28 | 496 |
| ornith:9b | 16 | 10 | 9 | 3.11 | 254 | 8,070 |
| gemma4:12b (app default) | 22 | 16 | 16 | 2.47 | 289 | 6,933 |

- **Qwen v2 vs v1:** +5 right size, but +1 right shape: most of the gain is parts with the right outline and a
  wrong or missing feature. Within noise on shape.
- **Granite v2:** the largest gain from fine-tuning (2 -> 10 right shape), level with ornith:9b and the best small
  model, at ~9x the speed of gemma4:12b (it writes no thinking text).
- **gemma4:12b** is still clearly ahead on shape (16). It cannot be fine-tuned on the 3080.
- Qwen v2 still writes ~2,400 output tokens per request (some thinking survives Ollama's Qwen template), which is
  why it is slower than Granite.

## Organic suite (SDF, 12 cases x 2 runs, the new prompt from #51; judged as above)

| model | built | best-effort mesh | nothing | built and judged right | mean judge score | s/case |
|---|---|---|---|---|---|---|
| **cad-qwen3.5:4b-v2** | **8** | 14 | 2 | **3** | 0.27 | 41 |
| cad-qwen3.5:4b-v1 | 2 | 12 | 10 | 0 | 0.03 | 85 |
| qwen3.5:4b (untuned) | 1 | 10 | 13 | 0 | 0.08 | 98 |
| cad-granite4.1:8b-v2 | 2 | 13 | 9 | 1 | 0.16 | 68 |
| granite4.1:8b (untuned) | 0 | 12 | 12 | 0 | 0.10 | 116 |
| gemini-3.5-flash-lite (teacher, 1 run) | 12/12 | | | 10/12 | 0.69 | |

- The SDF data works: Qwen v2 builds 4x as often as v1 and fails outright 5x less, and is the first student to
  sculpt objects the judge accepts (3 of 24). It is still far from the teacher.
- Granite learned much less from the same data.

## Gate and general replies (`finetune/general_check.py`, 3 runs of 10 gate + 5 general prompts)

| model | gate JSON | gate scope | general | CadQuery leaks |
|---|---|---|---|---|
| cad-qwen3.5:4b-v2 | 30/30 | 29/30 | 14/15 | 0/15 |
| cad-qwen3.5:4b-v1 | 30/30 | 27/30 | 14/15 | 0/15 |
| qwen3.5:4b | 30/30 | 28/30 | 14/15 | 0/15 |
| cad-granite4.1:8b-v2 | 29/30 | 28/30 | 14/15 | 0/15 |
| granite4.1:8b | 30/30 | 28/30 | 15/15 | 0/15 |

No narrowing: the students answer the gate and general prompts as well as their bases.

## Conclusions

- **Best student overall: cad-qwen3.5:4b-v2** (most right-size parts, the only one that sculpts, gate intact).
  **Best on CadQuery shape and speed: cad-granite4.1:8b-v2.** Neither reaches gemma4:12b on shape.
- More data mostly raised build and size rates; shape (the judge) moved less. Training on judged CadQuery runs, as
  the SDF data already is, is the obvious next filter.
- The next step with the most room is a stronger base: Gemma 4 12B (16 right shape untuned) on a rented 24 GB GPU,
  with the same data and code (`train.py --base unsloth/gemma-4-12b-it`).

## Caveats

- 36 main runs and 24 organic runs per model: differences of a few runs are noise (`rag-copying.md`).
- The judge is gemini-3.1-flash-lite, also the filter for the SDF training data, so organic scores may favour what
  that judge likes.
- Round 1's 591 CadQuery conversations still carry the benchmark-style caveat; the 231 new ones do not.
