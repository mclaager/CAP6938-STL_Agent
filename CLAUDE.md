# STL Generating Agent

CAP 6938 (AI Knowledge Systems, Fall 2026) course project. A chat agent that turns a
natural-language request into an STL file: the LLM writes CadQuery code, which is run in a
sandbox, validated as a mesh, and repaired in a loop. Planned work is tracked as GitHub issues
(`gh issue list`), not in a separate document.

## Environment

Managed with pixi. Everything runs in the `cap6938` environment:

- `pixi run -e cap6938 app` — Gradio UI (`app.py`) at http://127.0.0.1:7860
- `pixi run -e cap6938 examples` — rebuild and verify `examples.json` (`make_examples.py`)
- `pixi run -e cap6938 sandbox-test` — sandbox smoke test (`sandbox.py`)

LLM backend is picked by `CAD_PROVIDER`: `anthropic` (default, needs `ANTHROPIC_API_KEY`) or
`gemini` (free tier, needs `GEMINI_API_KEY`). `agent.py` reads a `.env` file in the project root;
it is gitignored, never commit keys. Optional: `CAD_MODEL`, `CAD_MAX_ATTEMPTS`, `CAD_TOP_K`
(see `agent.py` docstring), and `CAD_GATE` (`llm` default, `off`, `jev` stub) /
`CAD_GATE_MAX_ROUNDS` for the intake gate. With Gemini, the gate runs on `gemini-3.6-flash` by
default (override with `CAD_GATE_MODEL`) so its checks use a different daily quota than builds. Other OpenAI-compatible providers (Groq, OpenRouter, Ollama) can be
added as one entry in `PROVIDERS` in `agent.py`.

Gemini free tier is about 20 requests/day per model, and each repair attempt is one request.
If one model's quota runs out, switch with `CAD_MODEL` (e.g. `gemini-3.6-flash`), which has its own quota.

Add dependencies to `pixi.toml` under the `cap6938` feature. `requirements.txt` mirrors the
pypi deps for non-pixi users, so keep the two in sync.

## Layout

- `agent.py` — pipeline: TF-IDF RAG over `examples.json` → LLM code generation → sandboxed
  execution → mesh validation → self-repair. Entry point `run_pipeline(request, log)`.
- `sandbox.py` — `run_cadquery` (isolated execution with timeout) and `validate_mesh` (trimesh checks).
- `gate.py` — intake gate run before generation: the LLM reports scope, a filled-in spec and
  unknown parameters (critical or not); fixed rules in `decide_from_report` choose proceed /
  clarify / out_of_scope. Each gate check is one LLM request. `python gate.py` checks the rules offline.
- `app.py` — Gradio chat front end; uses `run_gate` (agent.py) for intake, then `run_pipeline` on
  the confirmed spec when the user clicks "Build it".
  Gradio's `Model3D` goes blank when updated from a streaming handler, so result panels are
  filled by a separate non-streaming `show_result` step.
  Layout CSS (fixed-height code/report, log stretched to line up with them) is passed via
  `demo.launch(**LAUNCH_KW)`; launching without it loses that layout.
- `make_examples.py` — source of truth for the few-shot example library; generates `examples.json`.
- `outputs/` — per-run folders plus each finished part as `<name>.stl` (name from the gate) and a
  colored `<name>.preview.glb` for the viewer (gitignored).
- `fonts/` — bundled Liberation Sans (SIL OFL). The sandbox uses it for any `text()` call without
  an explicit `fontPath`, because CadQuery's default "Arial" is missing on Linux and renders as boxes.

## Roadmap

Done: CadQuery + LLM pipeline, few-shot library + RAG, intake gate and chat UI.
Open work is in GitHub issues, with the original course timeline's target dates:

- #7 Evaluation harness: benchmark set + RAG ablation (weeks 3–4, overdue)
- #8 LLM-as-judge: render the mesh from several angles, score with a multimodal model (weeks 5–6, by 11/1)
- #9 Thingiverse fallback for requests code can't build (weeks 7–8, by 11/15)
- #10 Combine downloaded and generated parts into one STL (weeks 7–8)
- #11 Final report (weeks 9–10)
- #12 Jev gate backend and gate evaluation (not from the original proposal)
