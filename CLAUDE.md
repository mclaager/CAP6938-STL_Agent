# STL Generating Agent

CAP 6938 (AI Knowledge Systems, Fall 2026) course project. A chat agent that turns a
natural-language request into an STL file: the LLM writes CadQuery code, which is run in a
sandbox, validated as a mesh, and repaired in a loop. Planned work is tracked as GitHub issues
(`gh issue list`), not in a separate document.

## Environment

Managed with pixi. Everything runs in the `cap6938` environment:

Run commands from the repo root (modules are run with `python -m`, no install step):

- `pixi run -e cap6938 app` — Gradio UI (`stl_agent/app.py`) at http://127.0.0.1:7860
- `pixi run -e cap6938 test` — pytest suite in `tests/` (no LLM calls)
- `pixi run -e cap6938 examples` — rebuild and verify `stl_agent/data/examples.json`
- `pixi run -e cap6938 sdf-examples` — rebuild and verify `stl_agent/data/sdf_examples.json` (organic shapes, #42)
- `pixi run -e cap6938 bench run [--model M] [--category simple] [--budget 18]` — benchmark (#7);
  `pixi run -e cap6938 bench report` prints markdown tables from `eval/results/`;
  `bench run --suite organic [--config sdf sdf-norag thingiverse]` runs the organic cases (#42)
- `pixi run -e cap6938 retrieval-eval` — offline retrieval scores (no LLM requests, seconds)
- `pixi run -e cap6938 references` — build every reference solution and check it against `eval/cases.json`
- `pixi run -e cap6938 gate-specs` — save the gate's specs for the casual retrieval queries (one gate request each)
- `pixi run -e cap6938 size-check replay --model M [--variant claim blind] [--budget 100]` — replay size-only
  rejections with each repair message (#23); `size-check report` prints the table
- `pixi run -e cap6938 judge run --gen-model M [--config rag norag]` — visual judge over benchmark parts (#8);
  `judge calibrate` (pick the pass threshold), `judge report`, `judge sheet` (writes `eval/renders/index.html`)

- `pixi run -e cap6938 settings` — every setting, its current value and where it came from

Settings live in `settings.toml` in the repo root (gitignored, per user; another file with `CAD_SETTINGS`).
On first run it is created from the built-in defaults, `stl_agent/data/settings.defaults.toml`, with every
setting commented out: uncomment a line to change it. Commented lines use the default, so default changes still
apply; settings added later are appended commented out (user lines are never changed). `.env` (gitignored)
holds only API keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, optional `THINGIVERSE_TOKEN`); other `CAD_*` variables are not read, and the
app lists leftover ones at startup. The eval commands take `--set key=value` (repeatable, that run only), e.g.
`bench run --set provider=ollama --set providers.ollama.model=qwen3.5:9b`.

Rules for code:
- Read settings when used (`config.settings()`), never into module constants or default arguments: new code
  takes `None` defaults and reads `settings()` at call time, so the app can change settings without a restart
  (`config.set_setting` / `unset_setting` edit the file; `pipeline.reset_backends()` drops cached backends).
- A new setting goes in `settings.defaults.toml` with a comment, and its type in `TOP_FIELDS` /
  `PROVIDER_FIELDS` in `config.py`; `tests/test_config.py` fails if they disagree.
- Models and addresses are per provider (`[providers.<name>]`: `base_url`, `model`, `gate_model`,
  `judge_model`, `rpm`), resolved by `llm.provider_config`. `PROVIDERS` in `llm.py` holds only what users don't
  change (which env var has the key). A new OpenAI-compatible provider (Groq, OpenRouter, LM Studio) is one
  `PROVIDERS` entry plus a `[providers.<name>]` table in the defaults file.

Providers (`provider`): `anthropic` (default, needs `ANTHROPIC_API_KEY`), `gemini` (free tier, needs
`GEMINI_API_KEY`) or `ollama` (local models, no key, no quota; default `gemma4:12b`, the best of three local models tried, see
`eval/results/`). Ollama's server
defaults to a 4096-token context and drops the start of longer prompts without an error; raise it with
`OLLAMA_CONTEXT_LENGTH` on the Ollama service. Other settings: `max_attempts`, `api_hints` (wrong-call hints in repair feedback, default off),
`size_feedback` (repair message when only the size is off: `measured` default, `ask`, `fix`; also in the settings panel), `top_k`, `retriever`, `gate`
(`llm`, `off`, `jev` stub), `gate_max_rounds`, `gate_picks`, `thingiverse_results`, `thingiverse_pick` (`manual`/`auto`),
`organic_builder` (`thingiverse` default, or `sdf`: sculpt organic objects with code, #42), and for the visual judge `judge_provider`,
`providers.<p>.judge_model` (Gemini default `gemini-3.1-flash-lite`, a different model from the one that built
most benchmark parts) and `judge_threshold`. The gate uses the build model unless `providers.<p>.gate_model` is
set (with Gemini, another model has its own daily quota, so gate checks don't use up builds).

Gemini free tier is about 20 requests/day per model, and each repair attempt is one request.
If one model's quota runs out, switch with `providers.gemini.model` (or `--model` on `bench`), e.g.
`gemini-3.6-flash`, which has its own quota. The free tier also caps each model at 15 requests/minute; the
client paces itself to `providers.gemini.rpm` (default 10; at 14, Google still counted up to 17 in a minute)
and waits out short rate-limit replies instead of failing.
`gemini-3.5-flash-lite` has a much larger daily quota (about 500) but is a weaker coder; see `bench report`.

Add dependencies to `pixi.toml` under the `cap6938` feature. `requirements.txt` mirrors the
pypi deps for non-pixi users; `tests/test_requirements.py` checks that both list the same packages.

## Layout

`stl_agent/` is the Python package:

- `generate.py` — prompts, response parsing, and the generate → execute → validate → repair loop
  (`generate_cad`, `GenResult`). No caching or saving: the benchmark calls it directly. `language="sdf"` uses
  `SYSTEM_PROMPT_SDF` (the whole `sdf.py` API is listed in it) and runs the script as SDF (#42); the caller passes a
  retriever over `data/sdf_examples.json`.
- `pipeline.py` — the app's entry points: `run_gate(conversation, rounds)`, `run_pipeline(request, log, name)`,
  `run_fetch`, `run_assembly`, with the LLM, gate and retrievers cached until `reset_backends()`.
  `sculpt_organic` (in `run_gate`, with `organic_builder = "sdf"`) turns an out-of-scope object with an
  `organic_spec` into a build (scope `organic`, `run_pipeline(..., language="sdf")`) and Thingiverse pieces with one
  into `sdf` pieces; it runs before the token check, so sculpting needs no Thingiverse token. A failed sculpted
  piece falls back to an automatic Thingiverse pick, then its `fallback_spec`; in the app, a failed sculpted part
  shows the Thingiverse search instead.
- `config.py` — paths, `.env` loading (API keys), and settings: `Settings`, `settings()`, the settings file
  (create, validate, `set_setting` / `unset_setting`), `--set` overrides. Defaults in `data/settings.defaults.toml`.
- `llm.py` — LLM backends (Anthropic, OpenAI-compatible for Gemini etc., `ScriptedLLM` for tests), `make_llm`. Every backend
  follows the `LLM` protocol: `complete(system, messages) -> str` plus `input_tokens`, `output_tokens`, `last_truncated`.

- `replies.py` — `parse_json_object`, used by the gate and the judge to read JSON from model replies.
- `retrieval.py` — RAG over `data/examples.json` (each example's request plus its `keywords`: generic
  shape/operation words). Retrievers selected by the `retriever` setting: `tfidf` (default, words) or `tfidf-char`
  (letter chunks, numbers stripped). A local embedding model and a similarity cutoff were tried and removed
  (no gain; see `eval/results/retrieval-lib26.md`). Adding a public dataset (Text-to-CadQuery) as extra examples was
  also tried and removed (`eval/results/external-rag.md`). Keep the app lightweight (#19): no PyTorch.
- `sandbox.py` — `run_cadquery` (isolated execution with timeout) and `validate_mesh` (trimesh checks).
  `language="sdf"` runs an SDF script instead: `import sdf` / `from sdf import *` resolve to `stl_agent.sdf`.
- `sdf.py` — signed distance function modeling for organic shapes (#42). fogleman/sdf's names and operators
  (`sphere`, `capsule`, `a | b`, `.k(r)` blends, `.translate`, `.orient`, `.mirror`...) on our own small library
  (fogleman/sdf is not on PyPI and pulls in four more packages). `to_mesh` uses manifold3d's `level_set` (always
  closed): level_set asks for one point at a time, so points are answered from 16³ blocks evaluated with numpy,
  on the grid level_set samples (each axis split into floor(length / edge) cells, measured, not documented); a
  point off that grid is evaluated alone, so a wrong grid is correct but ~10x slower (a test checks). Where two
  parts just touch, an STL reader merges two vertices into a non-watertight pinch, so it retries on a shifted grid.
- `api_check.py` — extra repair feedback when a script fails (`api_hints` setting): reads the script with `ast`
  and checks each CadQuery call it can resolve against the installed CadQuery (`inspect`): missing methods (with
  close real names), arguments that don't fit the signature, a tuple where a number goes. Only used after a
  failed run, so a wrong guess can't block working code; `tests/test_api_check.py` checks it raises nothing on
  the examples and reference solutions.
- `gate/` — intake gate run before generation. The LLM reports scope, a filled-in spec and unknown
  parameters (critical or not); fixed rules in `base.decide_from_report` choose proceed / clarify /
  out_of_scope. `llm_gate.py` is the LLM backend, `jev.py` a stub (#12). Each gate check is one LLM request.
  Scope `composite` (#10, joined objects where one is sculpted) adds a `pieces` plan, checked by
  `base.parse_pieces`: base first, each piece `cadquery` (spec) or `thingiverse` (1-3 search phrases plus a
  `fallback_spec`, and an `organic_spec` for sculpting it), `height_mm`, and `place` (`on` an earlier piece, relation `on_top`/`under`/`beside`/`inside`).
  Fewer than 2 valid pieces falls back to `in_scope` or `out_of_scope`. For an out-of-scope object the gate also
  writes `organic_spec` (what it looks like, for sculpting; empty for logos, pictures and non-objects).
- `sources/thingiverse.py` — Thingiverse search and download (#9), plain httpx, Bearer app token
  (`THINGIVERSE_TOKEN`, optional: without it `pipeline.run_gate` turns plans with Thingiverse pieces into
  `out_of_scope`, so the app declines them as before #9/#10). `search(term)` returns
  `Candidate`s (author, license, `allows_derivatives`, thumbnail; NSFW dropped; count from
  `thingiverse_results`); `download_stl` takes the thing's largest STL via `download_url` (302 to the
  CDN). Cloudflare 403s default User-Agents (curl, httpx) on both API and CDN, so the client sends
  `USER_AGENT`. Tests use `httpx.MockTransport` (`client=` argument), never the network.
  `pipeline.run_fetch` saves it to `outputs/<name>.stl` plus `<name>.source.json` (attribution) and runs
  `validate_mesh`. The gate's JSON has a `search` field (keywords) for this.
- `sources/assembly.py` — combined parts (#10), pure trimesh: `place` applies the gate's relations using
  bounding boxes, dropping pieces straight down onto the surface below (`surface_heights`, numpy; trimesh's
  ray module would need rtree) and sinking each `OVERLAP` mm so they fuse; `merge` repairs open meshes and
  unions with manifold3d, else joins the pieces as overlapping bodies with a note. `pipeline.run_assembly`
  fetches Thingiverse pieces first (a missing model costs no build requests; none found → `fallback_spec` built
  with CadQuery), then generates, scales downloaded pieces and stand-ins to `height_mm`, places, merges.
- `render.py` — STL to one 768x512 PNG (3x2 grid) of six labelled views, offscreen VTK (works headless via EGL):
  - flat, straight-on: front, right, top, bottom (a through-hole shows the white background)
  - angled 3D: iso front, iso back (overall shape, and the back side)

  `RENDER_VERSION` is recorded with judged rows; bump it when views change, and add a judge prompt version that
  describes the new views. v2 differs from v1 in one view only: v1's angled "from below" view became the flat
  "bottom" view, because the judge read through-holes as blind pockets in the angled underside view.
- `judge/` — visual judge (#8), evaluation only (not run by the app). Same shape as `gate/`: `base.py`
  (`Judgement`, 0-1 score), `llm_judge.py` (one request with the request text, the measured size and the
  render). System prompts are files in `judge/prompts/<version>.txt`, never edited once used (results record
  the version); each is paired with the render version it describes (`RENDER_FOR_PROMPT`, `prompts/README.md`),
  and `eval/judge.py` warns on a mismatch. A Jev backend (#12) would be `judge/jev.py`. Images go to LLMs as message
  parts `{"type": "image", "png": bytes}`, converted per backend in `llm.py`.
- `app.py` — Gradio chat front end; uses `run_gate` for intake, then `run_pipeline` on the confirmed
  spec when the user clicks "Build it". An out-of-scope reply searches Thingiverse and shows the results
  in a gallery; clicking one runs `run_fetch` (`on_pick`, first in its event chain since only the first
  handler gets the click data) and hides the gallery. With `thingiverse_pick = "auto"` there is no gallery:
  the first result that downloads is used (`fetch_model`). A composite plan builds with `run_assembly`; with `thingiverse_pick = "manual"`,
  Build first shows a gallery per Thingiverse piece (`next_pick`, `pick_piece`), and `on_assemble`, chained
  after `on_pick`, builds once the last piece is picked. The settings sidebar (⚙) is in `settings_panel.py`.
  Gradio's `Model3D` goes blank when updated from a streaming handler, so result panels are
  filled by a separate non-streaming `show_result` step.
  Layout CSS (fixed-height code/report, log stretched to line up with them) is passed via
  `demo.launch(**LAUNCH_KW)`; launching without it loses that layout.
- `settings_panel.py` — the settings sidebar, built and wired by `build_panel(demo, cog)`. Its Thingiverse section
  sets `thingiverse_pick` (saved on change, `save_pick`) and shows whether `THINGIVERSE_TOKEN` is set; the Organic
  shapes and Builds sections set `organic_builder` and `size_feedback` (radio buttons saved on change, `save_choice`). Model settings
  picks provider, model, gate model and server address: the model dropdown lists the server's chat models
  (`llm.list_models`); "Test connection" sends one tiny request without saving (`make_llm(timeout=90, quick=True)`:
  no retries, so a busy model fails fast); "Save" writes the settings file (values equal to the default stay
  commented out) and calls `reset_backends()`, so the next request uses them.
- `data/examples.json` — few-shot library, generated by `scripts/make_examples.py` (the source of truth; edit that, not the JSON).
- `data/sdf_examples.json` — SDF few-shot library (11 organic objects), generated by `scripts/make_sdf_examples.py`,
  which checks each is one watertight solid standing on z = 0. None is an organic benchmark subject (a test checks).
- `data/fonts/` — bundled Liberation Sans (SIL OFL). The sandbox uses it for any `text()` call without
  an explicit `fontPath`, because CadQuery's default "Arial" is missing on Linux and renders as boxes.

Elsewhere:

- `eval/` — benchmark harness (#7). `cases.json` holds the requests (none copied from the example
  library; each has `expect` build/decline/either and an optional expected bounding box). `bench.py run`
  calls `generate_cad` directly (no gate), appends one line per (case, config) to
  `eval/results/<model>.jsonl` (committed), skips pairs already done, and stops before exceeding
  `--budget` requests, so a full run can be spread over several days of free-tier quota.
  `organic_cases.json` (#42, `--suite organic`): organic requests with no expected size, scored by the visual
  judge; configs `sdf` / `sdf-norag` (SDF scripts with or without the SDF examples; rows record the SDF library's
  id) and `thingiverse` (the first Thingiverse model that downloads for the case's `search`, no LLM requests).
  `--repeats N` runs each pair N times (LLM output varies run to run; with 3 repeats a one-run
  difference on a single case is still noise). Configs: `rag`, `norag`, `rag-hints` (rag plus `api_hints`; the others pin it off, as all
  earlier rows had it off), `rag-k1` / `rag-k5` (1 or 5
  examples instead of 3), `rag-char` (letter-chunk retriever), `rag-size` (the size-failure message shipped with #23; the other
  configs are pinned to the old one, `size_feedback = "fix"`, so earlier rows stay comparable). Older results also hold `rag-t20` / `rag-t30` rows (removed cutoff configs) and `rag-ext` rows (built-in library plus a Text-to-CadQuery pool, removed: no gain, `eval/results/external-rag.md`). Each result row records
  the example-library version (`library`); no-RAG rows are shared across library versions.
- `eval/judge.py` — judge CLI. Results in `eval/results/judge/<judge-model>.jsonl`, one row per (part, judge,
  prompt, render settings), so changing `--judge-model`, `--prompt` or `--px` adds a study beside the old
  one. Renders are committed under `eval/renders/<render version>-<px>px/` (~35 KB each), with a review/labelling page
  (`judge sheet`). `calibrate` judges each reference solution against its own request (should pass) and
  another case's request (should fail), plus hand labels in `eval/judge_labels.json`. Labels are graded on the
  judge's four bands (match / minor / feature wrong / different object; the top two pass), so calibration
  also reports band agreement (exact, within one band, lenient vs harsh), and `report` shows band counts.
- `eval/references.py` — hand-written CadQuery reference solutions for benchmark cases with a fixed size
  (`python -m eval.references`; also run by the tests). Add one for every new case with an expected bbox,
  so the case is known to be buildable and its bbox is right. Never shown to the model.
- `eval/size_check.py` — size-check replay (#23): rebuilds saved benchmark conversations up to a size-only
  rejection, sends each repair message variant (one request, or two for `blend`), runs the reply and records
  whether the part is accepted and at the right size. Results in `eval/results/size-check/<model>.jsonl`,
  write-up in `eval/results/size-check.md`. Its message texts are frozen; `tests/test_size_check.py` checks the
  shipped ones in `generate.build_size_message` still match.
- `eval/retrieval.py` + `eval/retrieval_labels.json` — offline retrieval evaluation (#20): hit@1, recall@3,
  MRR and latency per retriever, against hand-labelled relevant examples (original 19 cases only).
- `tests/` — gate rules, sandbox smoke tests, app chat flow, benchmark harness, retrieval (all offline).
- `scripts/` — maintenance scripts (`make_examples.py`, `make_sdf_examples.py`).
- `outputs/` (gitignored) — finished parts at the top: `<name>.stl` (name from the gate), a colored
  `<name>.preview.glb` for the viewer, and `<name>.source.json` for downloads. Working folders (each attempt's
  script and STL, `report.json`) go in `outputs/runs/` (`run_<time>`, `assembly_<time>`); benchmark runs in
  `outputs/bench/`.

Where planned work goes: Jev in
`stl_agent/gate/jev.py` and `stl_agent/judge/jev.py` (#12); the gate evaluation set goes in `eval/`.

## Roadmap

Done:
- CadQuery + LLM pipeline, few-shot library + RAG, intake gate and chat UI
- #7 Evaluation harness: 48-case benchmark with reference solutions, RAG and top-k ablation
  (`eval/results/ablation-lib26.md`)
- #8 LLM-as-judge: six-view renders scored by a multimodal model, calibrated against reference parts and
  hand-graded labels (`eval/results/judge-gemini-3.1-flash-lite.md`)
- #20 Retrieval evaluation (offline); retrievers that showed no gain were removed
- #9 Thingiverse fallback: out-of-scope requests search Thingiverse and download the picked model
  (searching on an explicit user request is not planned)
- #10 Combined parts: the gate plans pieces (generated or downloaded), placed by relation and merged into one STL
- #19 Fully local mode: Ollama provider, chosen in the app's settings panel (`eval/results/judge-local-gemma4-12b.md`)
- #23 Size check: a size-only failure says the script or the model's BBOX estimate may be wrong, with the measured
  size (`size_feedback`, `eval/results/size-check.md`)

Open work is in GitHub issues, with the original course timeline's target dates where there is one:

- #11 Final report (weeks 9–10)
- #12 Jev gate backend and gate evaluation (not from the original proposal; a Jev judge backend could follow)
- #28 Compare a stronger judge model on the hand-labelled parts
- #29 Test whether RAG makes the model copy the retrieved example instead of building every requested feature
- #35 Run the visual judge during builds (orientation of downloads, wrong-shape repair), as a user toggle
- #36 Edit a built part (transform, change the previous script, or rebuild from scratch)
- #39 Check that a downloaded model is the requested object (name/tag check on search results)
- #40 Show each piece of a combined part in the UI (per-piece code and STL downloads)
- #41 Copies of a piece in a combined part (`copy_of`, spread placement, distinct models when asked)
- #42 Build organic shapes with SDF modeling (own small library meshed by manifold3d, `organic_builder` setting)
- #46 Fine-tune a small local model on verified CadQuery/SDF builds (teacher runs filtered by the mesh checks and judge)
