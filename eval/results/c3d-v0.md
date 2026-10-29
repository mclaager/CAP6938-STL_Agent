# C3D-v0: an existing CadQuery fine-tune (#46)

Before training our own small model (#46), we tested the one public CadQuery fine-tune that runs in Ollama:
[numinousmuses/C3D-v0](https://huggingface.co/numinousmuses/C3D-v0). It is a LoRA of Gemma 3n E4B (4B),
trained for 1 epoch on about 48K Text-to-CadQuery examples (half the set). The only GGUF is Q8_0
(`hf.co/numinousmuses/C3D-v0-gguf:Q8_0`, 7.3 GB), run here as `c3d-v0:q8_0`. The model card gives no benchmark numbers.

Same 12 cases and repeats as the other local models (`eval/results/gemma4:12b.jsonl` etc.): `rag` and `rag-hints`,
3 runs each, plus `norag` (C3D-v0 was trained without examples in the prompt). RTX 3080, Ollama context 16K.

| model | config | n | correct | built | bbox ok | first try | avg attempts | s/case | output tokens/request |
|---|---|---|---|---|---|---|---|---|---|
| c3d-v0:q8_0 | rag | 36 | 8 | 10 | 5 | 7 | 3.36 | 25 | 531 |
| c3d-v0:q8_0 | rag-hints | 36 | 8 | 13 | 6 | 9 | 3.08 | 21 | 493 |
| c3d-v0:q8_0 | norag | 36 | 3 | 4 | 1 | 3 | 3.69 | 32 | 651 |
| qwen3.5:9b | rag | 36 | 9 | 9 | 9 | 3 | 3.56 | 84 | 1811 |
| ornith:9b | rag | 36 | 16 | 17 | 15 | 9 | 3.11 | 254 | 8070 |
| gemma4:12b | rag | 36 | 22 | 22 | 20 | 16 | 2.47 | 289 | 6933 |

(`correct` = built, and at the expected size where the case has one. `bbox ok` counts only cases with an expected size.)

## Findings

- **Weaker than every general local model with our prompt** (8/36 vs 9/36 for qwen3.5:9b, 22/36 for gemma4:12b),
  but about 10x faster than gemma4:12b because it writes no thinking text (~500 output tokens per request vs ~7,000).
- **Without examples it nearly fails** (3/36). The examples carry most of what it gets right.
- **Simple cases only:** its correct builds are almost all on s03, s04, m02, m03 and the casual c03. h10, m05, m08
  and m15 were never built in 9 tries each.
- **Errors:** most failed attempts are script errors (95 of 111 with `rag`). The most common one (23 attempts, all
  configs) is the PLAN/BBOX lines
  written *inside* the code block, a format it was never trained on. Next come invented or misused CadQuery
  arguments (`rect(..., startZ=)`, `extrude(centered=)`, `polygon()` without a diameter, `cboreHole(cbore_d=)`) and
  undefined variable names.
- **By hand, with a bare request (its training format):** one reply was a JSON sketch instead of code, one described the
  part's dimensions in prose, and the code it did write doubled every size (`cube_size = 25 * 2`) and called
  `cq.Pressure`, which does not exist.

## What this means for #46

- A fine-tune on Text-to-CadQuery alone doesn't make a good student for our requests. It matches the earlier
  finding that only 2.4% of Text-to-CadQuery scripts built at the stated size (`external-rag.md`): training data that
  was never run and size-checked teaches wrong sizes and wrong API calls.
- Its speed shows what a fine-tune can win: short replies in a fixed format. Our plan trains on teacher runs that were
  built and measured, in our PLAN/BBOX reply format, with the retrieved examples in the prompt.
