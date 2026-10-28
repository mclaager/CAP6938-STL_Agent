# Size check: repair message for a size-only failure (#23, 2026-10-28)

Question: `generate_cad` checks each part against the BBOX line the model writes, not against the request. When
that is the only problem (one watertight solid, wrong size), either the script or the model's own estimate can be
wrong. The old repair message always said "fix the script", so correct parts were rebuilt or lost. What should the
message say?

**Answer:** say that either can be wrong, give the measured size, and let the model decide. This is the new
`size_feedback = "measured"` default. After one repair, 53 of 90 parts are accepted at the right size, against 31
with the old message. 11 are accepted at the wrong size, against 5. Of the accepted parts, 17% are the wrong size,
against 14% before, and fewer parts need more repair requests. The other messages stay available as settings and
in the app's settings panel (Builds).

## Method
`eval/size_check.py` (`python -m eval.size_check replay --model M --variant ...`, then `report`):

- **Attempts:** every saved benchmark attempt (`outputs/bench/`, example library lib26-cdabac) that built one
  watertight body and failed only the size check, on a case with an expected size. Runs with `rag-hints` (other
  repair messages) and `rag-ext` (removed) are left out.
- **Count:** 90 attempts, from five source models: gemini-3.5-flash-lite 42, gemini-3.1-flash-lite 16, qwen3.5:9b
  15, gemma4:12b 13, ornith:9b 4.
- **Right size or not:** "right size" means the built part already matched the case's expected size (bench
  tolerance), so the BBOX line was what was wrong. That is 42 attempts. The other 48 were built at the wrong size.
- **Replay:** each conversation is rebuilt up to the rejection, one repair message is sent, and the reply's script
  is run and checked the way `generate_cad` would. One model answers every replay (gemini-3.5-flash-lite), whatever
  model made the attempt.
- **Most attempts by case:** m15 hex bolt (21), m16 star (19), h10, h05, h07, s03 hex.
  - Several of these are shapes whose overall size is easy to misjudge: the star's points don't reach the full
    width on both axes (claimed 60 x 60, built 57.06 x 54.27), and a hexagon is wider across corners than across
    flats.

## Messages
| variant | setting | text |
|---|---|---|
| current | `fix` | "The code ran, but the resulting mesh failed validation: Bounding box ... does not match expected ... Fix the script." |
| claim | `measured` | "...its size does not match your BBOX line: the part measures A mm, your BBOX says B mm. Either the script or the BBOX line is wrong. Work out from the request what the finished part's overall size should be. If the script already builds what the request asks for, reply with the same script unchanged and a corrected BBOX. Otherwise fix the script." |
| blind | `ask` | The same, without the measured size. |
| blend | (not shipped) | Blind first; if the reply fails on size alone again, claim. |

An earlier draft added a hint about stars and hexagons. It was dropped before any run, since it would fit this
benchmark's shapes rather than the problem.

## Results (gemini-3.5-flash-lite)
| variant | part was | n | accepted, right size | accepted, wrong size | still rejected / failed | same code | requests |
|---|---|---|---|---|---|---|---|
| current | right size | 42 | 14 (33%) | 4 (10%) | 24 | 4 | 1.00 |
| current | wrong size | 48 | 17 (35%) | 1 (2%) | 30 | 0 | 1.00 |
| claim | right size | 42 | 40 (95%) | 1 (2%) | 1 | 24 | 1.00 |
| claim | wrong size | 48 | 13 (27%) | 10 (21%) | 25 | 16 | 1.00 |
| blind | right size | 42 | 22 (52%) | 0 (0%) | 20 | 19 | 1.00 |
| blind | wrong size | 48 | 6 (12%) | 1 (2%) | 41 | 16 | 1.00 |
| blend | right size | 42 | 42 (100%) | 0 (0%) | 0 | 21 | 1.45 |
| blend | wrong size | 48 | 17 (35%) | 18 (38%) | 13 | 21 | 1.88 |

"same code": replies that kept the script unchanged in the first step. The two kinds of part together, out of 90:

| variant | accepted, right size | accepted, wrong size | wrong share of accepted | still failing (more attempts) |
|---|---|---|---|---|
| current (`fix`) | 31 | 5 | 14% | 54 |
| **claim (`measured`)** | **53** | **11** | **17%** | **26** |
| blind (`ask`) | 28 | 1 | 3% | 61 |
| blend | 59 | 18 | 23% | 13 (1.7 requests each) |

## Findings
- **The old message loses correct parts.** Told to fix a part that was already right, the model rewrote it: only
  a third came back accepted, and 4 of 42 came back at a wrong size.
- **Showing the measured size recovers almost every correct part** (40 of 42), mostly by keeping the script and
  correcting BBOX (24 kept the code).
- **It also lets the model copy the measured size into BBOX for a wrong part.** That happened in 10 of 48. This is
  the cost of the claim message.
- **Without the measured size (blind),** almost no wrong part gets through. But the model fixes fewer wrong parts,
  12% against the old message's 35%, and recovers only half the correct ones.
- **Blend was the worst on wrong parts.** Wrong parts mostly fail the blind step. The second step then shows them
  the measured size, so blend concentrates the copying on exactly those parts (38% let through).

## Limits
- One repair step, replayed by one model. In a real build, parts still failing get further attempts. The bench
  config `rag-size` (`bench run --config rag-size`) runs full builds with the new message, to compare with `rag`
  end to end. The other bench configs are pinned to the old message, so earlier rows stay comparable.
- "Right size" uses the cases' expected sizes. A part at the wrong size for the case may still be a reasonable
  reading of the request.
