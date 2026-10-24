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
pixi run -e cap6938 judge run --gen-model gemini-3.5-flash-lite   # visual judge over benchmark parts
pixi run -e cap6938 judge sheet              # review page: eval/renders/index.html
pixi run -e cap6938 retrieval-eval           # offline retrieval scores (no API calls)
```

## How it works

1. **Intake gate** (`stl_agent/gate/`): checks whether the request is in scope and whether any
   critical detail is missing; asks up to 3 questions, then shows the spec it will build.
2. **Retrieval** (`stl_agent/retrieval.py`): picks similar solved examples from `stl_agent/data/examples.json`.
3. **Generation and repair** (`stl_agent/pipeline.py`): the LLM writes CadQuery code; it runs in
   the sandbox (`stl_agent/sandbox.py`), the mesh is validated, and errors are fed back for another attempt.
4. **UI** (`stl_agent/app.py`): Gradio chat with a 3D preview, the generated code, and a validation report.

Settings (model, attempts, gate behaviour) are environment variables documented in `stl_agent/config.py`.

## Evaluation

- **Benchmark** (`eval/`): 48 requests from simple parts to out-of-scope ones, each fixed-size case with a
  hand-written reference solution. Results and the RAG ablation are in `eval/results/`.
- **Visual judge** (`stl_agent/judge/`, `eval/judge.py`): renders each generated part from six views and asks
  a multimodal model whether it matches the request, catching parts with the right size but the wrong shape.
  Calibrated against reference parts and hand-graded labels; renders are kept in `eval/renders/`.

## Roadmap

Done: the generation pipeline with retrieval and intake gate, the evaluation benchmark with RAG ablation,
and the visual judge. Open work is tracked in [GitHub issues](../../issues): Thingiverse fallback,
combining parts, the final report, a Jev gate backend, a fully local mode, and the size-check fix.
