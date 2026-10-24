# RAG ablation (issue #7)

gemini-3.5-flash-lite, example library lib26-cdabac, all 48 benchmark cases (`eval/cases.json`), up to
4 attempts each, run on 2026-10-24 with `python -m eval.bench run --config rag norag rag-k1 rag-k5`.

Every config is compared on **one run per case** (rep 0). `bench report` pools every repeat, and the
original 19 cases have 3 runs for rag/norag while the 29 newer cases have 1. That weights the easier
original cases three times, so its rag/norag totals differ from the table below.

## Result

| config | examples | correct | first try | avg attempts | requests | input tokens / case | output tokens / case |
|---|---|---|---|---|---|---|---|
| norag | 0 | 73% (35/48) | 33% | 2.21 | 106 | 3240 | 1239 |
| rag-k1 | 1 | 83% (40/48) | 56% | 1.77 | 85 | 2622 | 808 |
| **rag (default)** | **3** | **85% (41/48)** | 50% | 1.77 | 85 | 3293 | 751 |
| rag-k5 | 5 | 79% (38/48) | 52% | 1.85 | 89 | 4357 | 808 |

Input tokens include every repair attempt, so a config needing fewer attempts can use fewer input tokens
even with more examples per prompt.

| category | cases | norag | rag-k1 | rag | rag-k5 |
|---|---|---|---|---|---|
| simple | 5 | 5 | 4 | 5 | 5 |
| medium | 17 | 12 | 17 | 15 | 15 |
| hard | 13 | 7 | 7 | 8 | 7 |
| proposal | 8 | 7 | 8 | 8 | 8 |
| casual | 5 | 4 | 4 | 5 | 3 |
| original 19 cases | 19 | 17 | 16 | 17 | 17 |
| newer 29 cases (#22) | 29 | 18 | 24 | 24 | 21 |

## Reading

- **RAG helps: +12 points** of correct parts (73% → 85%). First-try success goes from 33% to 50%, and it
  uses 20% fewer requests (106 → 85), which matters on the free tier.
- **The gain comes entirely from the newer, harder cases** (18 → 24 of 29). On the original 19, rag and
  norag tie at 17/19 for this run. That set was close to saturated, which is why the cases were added. The
  earlier "+10 points" figure (3 repeats on those 19) holds up on the larger set, but the old set alone can
  no longer show it.
- **Top-k: keep 3.** k=1 is within one case of k=3 (83% vs 85%, within noise) and uses 20% fewer input
  tokens, so it's a reasonable cost saving if tokens matter. k=5 is worse (79%) and costs 32% more input
  tokens than k=3: extra, less relevant examples don't help this model.
- Declines (proposal) are handled correctly by every RAG config. Hard cases stay the weak spot: 7–8 of 13
  whatever the config.

## Caveats

- One run per case: a one-case difference is about 2 points and is noise. The RAG vs no-RAG gap (6 cases)
  is the only difference here large enough to rely on.
- The rep-0 rows for the original 19 cases (rag, norag) were run before the sphere/cone validation fix (#22),
  which mainly affects s05. rag-k1 and rag-k5 ran after it.
- One model only. Results for gemini-3.1-flash-lite (RAG only, no ablation) are in
  `gemini-3.1-flash-lite.jsonl`.
