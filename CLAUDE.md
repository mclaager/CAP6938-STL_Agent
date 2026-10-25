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
- `pixi run -e cap6938 bench run [--model M] [--category simple] [--budget 18]` — benchmark (#7);
  `pixi run -e cap6938 bench report` prints markdown tables from `eval/results/`
- `pixi run -e cap6938 retrieval-eval` — offline retrieval scores (no LLM requests, seconds)
- `pixi run -e cap6938 judge run --gen-model M [--config rag norag]` — visual judge over benchmark parts (#8);
  `judge calibrate` (pick the pass threshold), `judge report`, `judge sheet` (writes `eval/renders/index.html`)

LLM backend is picked by `CAD_PROVIDER`: `anthropic` (default, needs `ANTHROPIC_API_KEY`),
`gemini` (free tier, needs `GEMINI_API_KEY`) or `ollama` (local models, no key, no quota; default
`qwen3.5:9b`, address from `OLLAMA_BASE_URL`). Ollama's server defaults to a 4096-token context and drops
the start of longer prompts without an error; raise it with `OLLAMA_CONTEXT_LENGTH` on the Ollama service. `stl_agent/config.py` reads a `.env` file in the repo root;
it is gitignored, never commit keys. Optional: `CAD_MODEL`, `CAD_MAX_ATTEMPTS`, `CAD_TOP_K`
(see `config.py` docstring), and `CAD_GATE` (`llm` default, `off`, `jev` stub) /
`CAD_GATE_MAX_ROUNDS` for the intake gate. With Gemini, the gate runs on `gemini-3.6-flash` by
default (override with `CAD_GATE_MODEL`) so its checks use a different daily quota than builds.
The visual judge uses `CAD_JUDGE_MODEL` (default `gemini-3.1-flash-lite`, a different model from the one that
built most benchmark parts), optionally `CAD_JUDGE_PROVIDER`, and `CAD_JUDGE_THRESHOLD`.
Other OpenAI-compatible providers (Groq, OpenRouter) can be
added as one entry in `PROVIDERS` in `stl_agent/llm.py`.

Gemini free tier is about 20 requests/day per model, and each repair attempt is one request.
If one model's quota runs out, switch with `CAD_MODEL` (e.g. `gemini-3.6-flash`), which has its own quota.
The free tier also caps each model at 15 requests/minute; the client paces itself to `CAD_RPM`
(default 10 for Gemini; at 14, Google still counted up to 17 in a minute) and waits out short rate-limit replies instead of failing.
`gemini-3.5-flash-lite` has a much larger daily quota (about 500) but is a weaker coder; see `bench report`.

Add dependencies to `pixi.toml` under the `cap6938` feature. `requirements.txt` mirrors the
pypi deps for non-pixi users, so keep the two in sync.

## Layout

`stl_agent/` is the Python package:

- `pipeline.py` — prompts, response parsing, and the generate → execute → validate → repair loop
  (`generate_cad`). App entry points `run_gate(conversation, rounds)` and `run_pipeline(request, log, name)`.
- `config.py` — paths, `.env` loading and `CAD_*` settings.
- `llm.py` — LLM backends (Anthropic, OpenAI-compatible for Gemini etc., `ScriptedLLM` for tests), `make_llm`.
- `retrieval.py` — RAG over `data/examples.json` (each example's request plus its `keywords`: generic
  shape/operation words). Retrievers selected by `CAD_RETRIEVER`: `tfidf` (default, words) or `tfidf-char`
  (letter chunks, numbers stripped). A local embedding model and a similarity cutoff were tried and removed
  (no gain; see `eval/results/retrieval-lib26.md`). Keep the app lightweight (#19): no PyTorch.
- `sandbox.py` — `run_cadquery` (isolated execution with timeout) and `validate_mesh` (trimesh checks).
- `gate/` — intake gate run before generation. The LLM reports scope, a filled-in spec and unknown
  parameters (critical or not); fixed rules in `base.decide_from_report` choose proceed / clarify /
  out_of_scope. `llm_gate.py` is the LLM backend, `jev.py` a stub (#12). Each gate check is one LLM request.
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
  spec when the user clicks "Build it".
  Gradio's `Model3D` goes blank when updated from a streaming handler, so result panels are
  filled by a separate non-streaming `show_result` step.
  Layout CSS (fixed-height code/report, log stretched to line up with them) is passed via
  `demo.launch(**LAUNCH_KW)`; launching without it loses that layout.
- `data/examples.json` — few-shot library, generated by `scripts/make_examples.py` (the source of truth; edit that, not the JSON).
- `data/fonts/` — bundled Liberation Sans (SIL OFL). The sandbox uses it for any `text()` call without
  an explicit `fontPath`, because CadQuery's default "Arial" is missing on Linux and renders as boxes.

Elsewhere:

- `eval/` — benchmark harness (#7). `cases.json` holds the requests (none copied from the example
  library; each has `expect` build/decline/either and an optional expected bounding box). `bench.py run`
  calls `generate_cad` directly (no gate), appends one line per (case, config) to
  `eval/results/<model>.jsonl` (committed), skips pairs already done, and stops before exceeding
  `--budget` requests, so a full run can be spread over several days of free-tier quota.
  `--repeats N` runs each pair N times (LLM output varies run to run; with 3 repeats a one-run
  difference on a single case is still noise). Configs: `rag`, `norag`, `rag-k1` / `rag-k5` (1 or 5
  examples instead of 3), `rag-char` (letter-chunk retriever). Older results also hold `rag-t20` / `rag-t30` rows (removed cutoff configs). Each result row records
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
- `eval/retrieval.py` + `eval/retrieval_labels.json` — offline retrieval evaluation (#20): hit@1, recall@3,
  MRR and latency per retriever, against hand-labelled relevant examples (original 19 cases only).
- `tests/` — gate rules, sandbox smoke tests, app chat flow, benchmark harness, retrieval (all offline).
- `scripts/` — maintenance scripts (`make_examples.py`).
- `outputs/` — per-run folders plus each finished part as `<name>.stl` (name from the gate) and a
  colored `<name>.preview.glb` for the viewer (gitignored).

Where planned work goes: Thingiverse and part combining in `stl_agent/sources/` (#9, #10), Jev in
`stl_agent/gate/jev.py` and `stl_agent/judge/jev.py` (#12); the gate evaluation set goes in `eval/`.

## Roadmap

Done:
- CadQuery + LLM pipeline, few-shot library + RAG, intake gate and chat UI
- #7 Evaluation harness: 48-case benchmark with reference solutions, RAG and top-k ablation
  (`eval/results/ablation-lib26.md`)
- #8 LLM-as-judge: six-view renders scored by a multimodal model, calibrated against reference parts and
  hand-graded labels (`eval/results/judge-gemini-3.1-flash-lite.md`)
- #20 Retrieval evaluation (offline); retrievers that showed no gain were removed

Open work is in GitHub issues, with the original course timeline's target dates where there is one:

- #9 Thingiverse fallback for requests code can't build (weeks 7–8, by 11/15)
- #10 Combine downloaded and generated parts into one STL (weeks 7–8)
- #11 Final report (weeks 9–10)
- #12 Jev gate backend and gate evaluation (not from the original proposal; a Jev judge backend could follow)
- #19 Fully local mode, configurable in the app
- #23 Size check rejects correct parts when the model's own BBOX estimate is wrong
- #28 Compare a stronger judge model on the hand-labelled parts
- #29 Test whether RAG makes the model copy the retrieved example instead of building every requested feature
