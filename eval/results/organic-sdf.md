# Organic shapes with SDF modeling (#42, 2026-10-28, interim)

Question: can the agent build organic objects (animals, plants, figurines) itself, by writing signed distance
function (SDF) scripts (`stl_agent/sdf.py`), instead of downloading them from Thingiverse?

**So far: it builds most of them, and they are recognisable but crude.** Only cases o01–o06 are judged so far.
- With the SDF examples, the visual judge passes 5 of 6 parts (mean score 0.67).
- Thingiverse also passes 5 of 6, but with a lower mean (0.54).
- To the eye, though, the Thingiverse models are far more detailed, except that its "tulip" is a rose.

So `organic_builder` stays `thingiverse` by default, and sculpting is an experimental option. It is useful when
there is no Thingiverse token, or no matching model.

## Setup
- **Cases:** `eval/organic_cases.json`, 12 organic requests written like the gate's `organic_spec` (what the
  object looks like, overall size). None is an object in the SDF example library (a test checks).
- **Model:** gemini-3.5-flash-lite, max 4 attempts, `size_feedback = "measured"`; one run per case.
- **Configs:**
  - `sdf`: SDF script, 3 retrieved SDF examples (library lib11-5cd03c).
  - `sdf-norag`: SDF script, no examples. The system prompt still lists the whole API.
  - `thingiverse`: the first model that downloads for the case's `search` words, unscaled (no LLM requests).
- **Judge:** gemini-3.1-flash-lite, prompt v2, renders v2. It sees the request, the measured size and six views,
  and passes a part at a score of 0.5 or more.

## Build results (all 12 cases)
| config | built (valid mesh) | best effort | first try | requests |
|---|---|---|---|---|
| sdf | 9 / 12 | 3 (o04, o05, o12) | 0 | 36 |
| sdf-norag | 10 / 12 | 2 (o09, o10) | 0 | 33 |
| thingiverse | 9 / 12 watertight | 3 not watertight (o04, o05, o08) | – | 0 |

- **No case passed on the first attempt.** Almost every first attempt failed the size check against the
  model's own BBOX line: organic sizes are hard to estimate from blended shapes. The size message from #23
  usually fixed it in one repair.
- **The other main failure was separate bodies** (2–3 disconnected pieces): a part placed so it doesn't touch
  the rest. Without examples, there were also more script errors (5 against 1).

## Judge results (o01–o06)
| config | judged | pass | mean score | match | minor | feature wrong | different object |
|---|---|---|---|---|---|---|---|
| sdf | 6 | 5 | 0.67 | 2 | 1 | 3 | 0 |
| sdf-norag | 6 | 4 | 0.52 | 2 | 1 | 1 | 2 |
| thingiverse | 6 | 5 | 0.54 | 0 | 3 | 2 | 1 |

![iso views of o01-o06, top row sdf, bottom row sdf-norag](organic-sdf-o01-o06.png)

Renders of the downloaded Thingiverse models are not committed: they are other people's models, many licensed
non-commercial and/or no-derivatives (`judge run` recreates them locally from `outputs/bench/`). Licenses of
the 12 downloads: 4 CC BY-SA, 3 CC BY-NC-ND, 2 CC BY-NC-SA, 1 CC BY-NC, 1 CC BY, 1 public domain.

What the renders and judge notes show:
- **Examples help.** Without them, two of six parts were featureless egg shapes, the cat and the horse:
  the main body covered every other feature.
- **With examples, the model sometimes copies an example's features.** The cat (o01) got the rabbit example's
  long ears. Whether retrieved examples crowd out what was asked is #29.
- **The judge is generous to crude shapes.** It gave 0.89 to blob-like frogs and owls that have the listed
  features (bulging eyes, ear tufts), and it docks Thingiverse models for size: downloads come at any scale,
  while the app scales them (combined parts) or the user does.
- **Thingiverse quality varies with the search.** "tulip flower" returned a rose, scored 0.

## Next
- Judge o07–o12 (about 18 judge requests, the judge model's daily quota).
- Repeat runs (one run per case is noisy).
- Disconnected parts are the most common failure. The repair message could say which parts float, using the
  mesh's bodies.
