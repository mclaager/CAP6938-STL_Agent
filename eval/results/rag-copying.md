# Does RAG make the model copy the retrieved example? (#29, 2026-10-28)

Question: some RAG failures in the ablation (#7) looked like the model copied a retrieved example and left out
a requested feature (the clamp ring built as a plain washer). Is that real, or one-run noise?

**Answer: yes, but it is small, and it cancels RAG's gain in size.**
- With examples, more parts come out the right size (106 vs 97), but twice as many of those have the wrong shape
  (26 vs 13).
- Parts that are both the right size and the right shape: 80 with RAG, 84 without. Per case, RAG is better on 8
  cases and worse on 10 (sign test p = 0.82).
- A prompt line asking for every requested feature (`rag-features`) recovered 2 of the copied parts (10 of 21 runs
  on the 7 copy-like cases, against 8 for RAG and 12 for no RAG). That is within noise, so it is not shipped.
- About 10 of RAG's 26 wrong shapes look copied: the missing feature is in none of the 3 retrieved examples, and
  the part looks like one of them. The three suspected cases (m08, m11, m17) are the clearest: there, no RAG
  passes 8 of 9 runs, RAG 4 of 9.

## Setup
- **Model:** gemini-3.5-flash-lite; configs `rag` (3 examples, library lib26-cdabac) and `norag`, as in the
  ablation (`size_feedback = "fix"`, no API hints, max 4 attempts).
- **Runs:** 3 per case on the 42 cases expected to build, 252 runs.
- **Judge:** gemini-3.1-flash-lite, prompt v2, renders v2, pass at score 0.5 (as in
  `judge-gemini-3.1-flash-lite.md`). A part counts when the benchmark's size check passes **and** the judge
  passes it; a part that didn't build counts as a failure.

## Results
| | rag | norag |
|---|---|---|
| runs | 126 | 126 |
| right size | 106 | 97 |
| right size, wrong shape (judge fail) | 26 | 13 |
| right size and shape | 80 | 84 |
| right size and shape, per repeat | 22 / 28 / 30 | 24 / 35 / 25 |

- **Run-to-run noise is large:** no RAG ranges from 24 to 35 between repeats. One run per case (the ablation)
  cannot separate the two configs; three runs show them level.
- **Cases where they differ** (successes out of 3): RAG better on c04 (2-1), h01 (3-2), h02 (1-0), h05 (1-0),
  h09 (2-1), m05 (3-2), m07 (3-2), m16 (2-0); worse on c01 (2-3), c03 (0-2), h03 (0-1), h10 (0-1), h12 (1-2),
  m08 (1-2), m09 (1-3), m10 (0-1), m11 (2-3), m17 (1-3).

## RAG parts of the right size with the wrong shape (26)
For each, the judge's problem and the 3 retrieved examples:

| kind | parts | what happened |
|---|---|---|
| **looks copied** (10) | m08 ×2, m11, m17 ×2, h04, h07, m12 ×2, p01 | clamp ring → the washer example (no gap); keyhole → the slot example (no round hole); arch block → the plain blocks (no arch); pulley → plain wheel like the gear blank (no V-groove); shaft collar → washer-like (no radial hole); elbow with straight ends beside the L-bracket; ruler with the name but almost no ticks, beside the engraved coaster |
| **not copied** (11) | c01, c03 ×3, c04 (rep 2), h03, h08, h12 (rep 0), m03, m14, p06 | the part is unlike the examples, or an example even has the missing feature (countersunk holes in h03, through-holes in h08, a floor in m14) |
| **judge error** (3) | m13 ×3 | the stepped shaft called "symmetrical", a known judge error (#28); no RAG fails it the same way |
| **unclear** (2) | c05, h12 (rep 2) | a desk hook with no upturned end beside a mounting strip; a thumbwheel without the hex socket beside a hex nut |

On the copied cases, no RAG does as well or better: m08 1 vs 2, m11 2 vs 3, m17 1 vs 3, h04 1 vs 1,
m12 1 vs 1, p01 2 vs 2, h07 0 vs 0 (RAG vs no RAG, of 3).

## A prompt line against copying (`rag-features`, 2026-10-29)
Config `rag-features` is `rag` plus one line after the examples (`generate.FEATURE_NOTE`): "The examples may lack
features this request asks for. Build every feature the request names (each hole, slot, gap, groove, step,
marking...), even if no example has it." Run on the 7 copy-like cases, 3 runs each (21 runs, 50 requests), same
model, settings and judge as above.

| case | rag | rag-features | norag |
|---|---|---|---|
| m08 clamp ring (gap) | 1 | 2 | 2 |
| m11 keyhole plate (round hole) | 2 | 2 | 3 |
| m17 arch block (arch) | 1 | 2 | 3 |
| h04 pulley (V-groove) | 1 | 1 | 1 |
| m12 shaft collar (radial hole) | 1 | 1 | 1 |
| p01 ruler (ticks) | 2 | 2 | 2 |
| h07 pipe elbow (no straight ends) | 0 | 0 | 0 |
| **right size and shape, of 21** | **8** | **10** | **12** |
| right size | 18 | 15 | 14 |
| requests | 45 | 50 | 50 |

Runs passing both checks, of 3 per case.

- **The line helps a little, where the missing feature is a single cut:** the clamp ring's gap and the arch each
  came out once more. It changed nothing on the other five cases.
- **+2 of 21 is within noise** (one repeat of no RAG alone moves by 11 parts over 42 cases), and the line still
  trails no RAG on these cases. It also cost size: 15 right-size parts against 18, since fewer parts built (16
  against 19).
- **Remaining failures are mostly not copying** but features the model can't build: the pulley's V-groove (2 of 3
  runs flat), the collar's radial hole, the elbow (solid, or a hemisphere). These fail without examples too.
- So it is not shipped: no setting, and the app's prompt is unchanged. The config stays for comparison.

## Next
- Since copying only explains part of the gap, a better lever is checking shape during the build: run the judge on
  the finished part and send its problems back as a repair message (#35).
