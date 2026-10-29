# Does RAG make the model copy the retrieved example? (#29, 2026-10-28)

Question: some RAG failures in the ablation (#7) looked like the model copied a retrieved example and left out
a requested feature (the clamp ring built as a plain washer). Is that real, or one-run noise?

**Answer: yes, but it is small, and it cancels RAG's gain in size.**
- With examples, more parts come out the right size (105 vs 94), but twice as many of those have the wrong shape
  (25 vs 12).
- Parts that are both the right size and the right shape: 79 with RAG, 81 without. Per case, RAG is better on 7
  cases and worse on 8 (sign test p = 1.0).
- About 10 of RAG's 25 wrong shapes look copied: the missing feature is in none of the 3 retrieved examples, and
  the part looks like one of them. The three suspected cases (m08, m11, m17) are the clearest: there, no RAG
  passes 8 of 9 runs, RAG 4 of 9.

## Setup
- **Model:** gemini-3.5-flash-lite; configs `rag` (3 examples, library lib26-cdabac) and `norag`, as in the
  ablation (`size_feedback = "fix"`, no API hints, max 4 attempts).
- **Runs:** 3 per case on the 42 cases expected to build. 5 of 252 runs are still missing (rep 2 of c02-c05, quota); the
  counts below are over the 247 done.
- **Judge:** gemini-3.1-flash-lite, prompt v2, renders v2, pass at score 0.5 (as in
  `judge-gemini-3.1-flash-lite.md`). A part counts when the benchmark's size check passes **and** the judge
  passes it; a part that didn't build counts as a failure.

## Results
| | rag | norag |
|---|---|---|
| runs | 125 | 122 |
| right size | 105 | 94 |
| right size, wrong shape (judge fail) | 25 | 12 |
| right size and shape | 79 | 81 |
| right size and shape, per repeat | 22 / 28 / 29 | 24 / 35 / 22 |

- **Run-to-run noise is large:** no RAG ranges from 22 to 35 between repeats. One run per case (the ablation)
  cannot separate the two configs; three runs show them level.
- **Cases where they differ** (successes out of 3): RAG better on h01 (3-2), h02 (1-0), h05 (1-0), h09 (2-1),
  m05 (3-2), m07 (3-2), m16 (2-0); worse on h03 (0-1), h10 (0-1), h12 (1-2), m08 (1-2), m09 (1-3), m10 (0-1),
  m11 (2-3), m17 (1-3).

## RAG parts of the right size with the wrong shape (25)
For each, the judge's problem and the 3 retrieved examples:

| kind | parts | what happened |
|---|---|---|
| **looks copied** (10) | m08 ×2, m11, m17 ×2, h04, h07, m12 ×2, p01 | clamp ring → the washer example (no gap); keyhole → the slot example (no round hole); arch block → the plain blocks (no arch); pulley → plain wheel like the gear blank (no V-groove); shaft collar → washer-like (no radial hole); elbow with straight ends beside the L-bracket; ruler with the name but almost no ticks, beside the engraved coaster |
| **not copied** (10) | c01, c03 ×3, h03, h08, h12 (rep 0), m03, m14, p06 | the part is unlike the examples, or an example even has the missing feature (countersunk holes in h03, through-holes in h08, a floor in m14) |
| **judge error** (3) | m13 ×3 | the stepped shaft called "symmetrical", a known judge error (#28); no RAG fails it the same way |
| **unclear** (2) | c05, h12 (rep 2) | a desk hook with no upturned end beside a mounting strip; a thumbwheel without the hex socket beside a hex nut |

On the copied cases, no RAG does as well or better: m08 1 vs 2, m11 2 vs 3, m17 1 vs 3, h04 1 vs 1,
m12 1 vs 1, p01 2 vs 2, h07 0 vs 0 (RAG vs no RAG, of 3).

## Next
- Try a line in the generation prompt (not the judge's) asking the model to build every requested feature
  even when the examples don't have one, as a `rag-*` config, on the copied cases first (h04, h07, m08, m11, m12,
  m17, p01; 3 runs each, ~40 requests), then on all cases if it helps.
- Finish the 5 missing runs.
