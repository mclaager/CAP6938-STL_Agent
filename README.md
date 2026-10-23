# STL Generating Agent

A chat agent that turns a plain-English part description into a 3D-printable STL file. An LLM
writes [CadQuery](https://cadquery.readthedocs.io) code, which runs in a sandbox, is checked as a
mesh, and is repaired in a loop when it fails. Before building, an intake step asks about missing
details that matter and declines shapes CadQuery can't build.

Course project for CAP 6938 (AI Knowledge Systems, Fall 2026).

## Run it

Requires [pixi](https://pixi.sh).

```bash
echo "GEMINI_API_KEY=your-key" > .env      # free key: aistudio.google.com
echo "CAD_PROVIDER=gemini" >> .env         # or ANTHROPIC_API_KEY=... for Claude (the default)
pixi run -e cap6938 app                    # http://127.0.0.1:7860
```

Without pixi: `pip install -r requirements.txt`, then `python -m stl_agent.app` from the repo root.

Other commands:

```bash
pixi run -e cap6938 test       # tests (no API calls)
pixi run -e cap6938 examples   # rebuild the few-shot example library
pixi run -e cap6938 bench run --budget 18   # benchmark; resumes where it stopped
pixi run -e cap6938 bench report             # results tables
```

## How it works

1. **Intake gate** (`stl_agent/gate/`): checks whether the request is in scope and whether any
   critical detail is missing; asks up to 3 questions, then shows the spec it will build.
2. **Retrieval** (`stl_agent/retrieval.py`): picks similar solved examples from `stl_agent/data/examples.json`.
3. **Generation and repair** (`stl_agent/pipeline.py`): the LLM writes CadQuery code; it runs in
   the sandbox (`stl_agent/sandbox.py`), the mesh is validated, and errors are fed back for another attempt.
4. **UI** (`stl_agent/app.py`): Gradio chat with a 3D preview, the generated code, and a validation report.

Settings (model, attempts, gate behaviour) are environment variables documented in `stl_agent/config.py`.

## Roadmap

Planned work is tracked in [GitHub issues](../../issues): evaluation harness, LLM-as-judge
benchmark, Thingiverse fallback, combining parts, and the final report.
