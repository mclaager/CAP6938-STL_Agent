# STL Generating Agent

CAP 6938 (AI Knowledge Systems, Fall 2026) course project. A chat agent that turns a
natural-language request into an STL file: the LLM writes CadQuery code, which is run in a
sandbox, validated as a mesh, and repaired in a loop. Full scope and timeline:
`CAP_6938_Project_Proposal-1.pdf`.

## Environment

Managed with pixi. Everything runs in the `cap6938` environment:

- `pixi run -e cap6938 app` — Gradio UI (`app.py`) at http://127.0.0.1:7860
- `pixi run -e cap6938 examples` — rebuild and verify `examples.json` (`make_examples.py`)
- `pixi run -e cap6938 sandbox-test` — sandbox smoke test (`sandbox.py`)

LLM backend is picked by `CAD_PROVIDER`: `anthropic` (default, needs `ANTHROPIC_API_KEY`) or
`gemini` (free tier, needs `GEMINI_API_KEY`). `agent.py` reads a `.env` file in the project root;
it is gitignored, never commit keys. Optional: `CAD_MODEL`, `CAD_MAX_ATTEMPTS`, `CAD_TOP_K`
(see `agent.py` docstring). Other OpenAI-compatible providers (Groq, OpenRouter, Ollama) can be
added as one entry in `PROVIDERS` in `agent.py`.

Gemini free tier is about 20 requests/day per model, and each repair attempt is one request.
If one model's quota runs out, switch with `CAD_MODEL` (e.g. `gemini-3.6-flash`), which has its own quota.

Add dependencies to `pixi.toml` under the `cap6938` feature. `requirements.txt` mirrors the
pypi deps for non-pixi users, so keep the two in sync.

## Layout

- `agent.py` — pipeline: TF-IDF RAG over `examples.json` → LLM code generation → sandboxed
  execution → mesh validation → self-repair. Entry point `run_pipeline(request, log)`.
- `sandbox.py` — `run_cadquery` (isolated execution with timeout) and `validate_mesh` (trimesh checks).
- `app.py` — Gradio front end; only depends on `run_pipeline`.
- `make_examples.py` — source of truth for the few-shot example library; generates `examples.json`.
- `outputs/` — per-run generated files (gitignored).
- `fonts/` — bundled Liberation Sans (SIL OFL). The sandbox uses it for any `text()` call without
  an explicit `fontPath`, because CadQuery's default "Arial" is missing on Linux and renders as boxes.

## Roadmap (from proposal)

1. Weeks 1–4: CadQuery + LLM setup, few-shot library + RAG (done / in progress)
2. Weeks 5–6: LLM-as-judge benchmark: render the mesh from several angles, score with a multimodal model
3. Weeks 7–8: Thingiverse API integration, so the agent can fetch existing STLs when generating with code isn't practical
4. Weeks 9–10: report and final improvements
