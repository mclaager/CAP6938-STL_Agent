# STL Generating Agent

A chat agent that turns a plain-English part description into a 3D-printable STL file. An LLM
writes [CadQuery](https://cadquery.readthedocs.io) code, which runs in a sandbox, is checked as a
mesh, and is repaired in a loop when it fails. Before building, an intake step asks about missing
details that matter and declines shapes CadQuery can't build.

Course project for CAP 6938 (AI Knowledge Systems, Fall 2026).

## Run it

Requires [pixi](https://pixi.sh).

```bash
echo "GEMINI_API_KEY=your-key" > .env      # free key: aistudio.google.com (or ANTHROPIC_API_KEY=... for Claude)
pixi run -e cap6938 app                    # http://127.0.0.1:7860
```

The first run creates `settings.toml` in the project root, listing every setting with its default, commented
out. To change one, uncomment its line and edit it: for example `provider = "gemini"` for the key above, or
`provider = "ollama"` to run on a local model with [Ollama](https://ollama.com) (no key needed).
Provider, model and server address can also be set in the app, under "Model settings" ("Test connection"
checks them first). `.env` holds only API keys. `pixi run -e cap6938 settings` shows every setting and where
its value came from.

Requests that can't be built with CAD code (statues, animals, characters) get a Thingiverse search instead,
if `.env` has a `THINGIVERSE_TOKEN`: register an app at <https://www.thingiverse.com/apps/create> and copy
its App Token. The results show as thumbnails; clicking one downloads its STL, with the author and license
shown in the chat and saved next to it (`outputs/<name>.source.json`). Thingiverse is optional: without a token, the app
declines sculpted parts, and parts that would join them with others, as it did before.

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
pixi run -e cap6938 references               # check the benchmark's reference solutions build
```

## Run it locally (no API key)

The agent can run entirely on your machine with [Ollama](https://ollama.com); nothing leaves the machine and
there is no daily quota.

1. Install Ollama and pull a model: `ollama pull gemma4:12b`.
2. Raise Ollama's context window. Its server defaults to 4096 tokens and silently drops the start of longer
   prompts, which breaks repair attempts. With the Linux service: `sudo systemctl edit ollama`, add
   ```ini
   [Service]
   Environment="OLLAMA_CONTEXT_LENGTH=16384"
   ```
   then `sudo systemctl restart ollama`. (Running `ollama serve` by hand: set the variable before it.)
3. In the app, open the settings panel (⚙, top right), pick `ollama`, check it with "Test connection", and
   Save. Or set `provider = "ollama"` in `settings.toml`. Ollama on another machine: change the server address
   (default `http://localhost:11434/v1`).

The intake gate uses the same local model. The visual judge (evaluation only) also runs locally with
`gemma4:12b` (`--set judge_provider=ollama --set judge_threshold=0.94`): about as accurate as Gemini on the
calibration set, but more lenient; see `eval/results/judge-local-gemma4-12b.md`.

**Which model.** Three models that fit a 10 GB GPU (RTX 3080), on a 12-case subset of the benchmark
(3 runs each, with retrieval; results in `eval/results/<model>.jsonl`):

| Model | Download | Correct parts | First try | Median time per case |
|---|---|---|---|---|
| `gemma4:12b` (default) | 8.0 GB | 22/36 (61%) | 16/36 | 207 s |
| `ornith:9b` | 5.6 GB | 16/36 (44%) | 9/36 | 260 s |
| `qwen3.5:9b` | 6.6 GB | 9/36 (25%) | 3/36 | 73 s |

For comparison, Gemini's free `gemini-3.5-flash-lite` got 23/27 (85%) of the same cases right. Local models
are slower and miss more on complex parts: `gemma4:12b` got all 6 simple runs and 14 of 18 medium ones right,
but only 2 of 9 hard ones. Expect a few minutes per part. `gemma4:12b` needs about
10 GB of GPU memory with a 16K context; with less, pick a smaller model or Ollama runs part of it on the CPU,
which is much slower. A reply that runs on can take over 5 minutes, so Ollama's request timeout is 600 s
(`providers.ollama.timeout`).

## How it works

1. **Intake gate** (`stl_agent/gate/`): checks whether the request is in scope and whether any
   critical detail is missing; asks up to 3 questions, then shows the spec it will build.
2. **Retrieval** (`stl_agent/retrieval.py`): picks similar solved examples from `stl_agent/data/examples.json`.
3. **Generation and repair** (`stl_agent/generate.py`): the LLM writes CadQuery code; it runs in
   the sandbox (`stl_agent/sandbox.py`), the mesh is validated, and errors are fed back for another attempt.
4. **Combined parts** (`stl_agent/sources/assembly.py`): for requests like "a cat sitting on a tree", the gate
   plans pieces, each downloaded from Thingiverse or built with CadQuery, plus where each one goes; they are
   scaled, placed and merged into one STL. In the settings panel (⚙), "Picking models" chooses whether you
   pick each model from a gallery or the first suitable result is used (`thingiverse_pick`).
5. **Thingiverse fallback** (`stl_agent/sources/thingiverse.py`): for out-of-scope requests, searches with
   keywords from the gate and downloads the model the user picks.
6. **UI** (`stl_agent/app.py`): Gradio chat with a 3D preview, the generated code, and a validation report.

Settings (model, attempts, gate behaviour) are in `settings.toml`; the defaults, with a comment for each, are in
`stl_agent/data/settings.defaults.toml`.

## Evaluation

- **Benchmark** (`eval/`): 48 requests from simple parts to out-of-scope ones, each fixed-size case with a
  hand-written reference solution. Results and the RAG ablation are in `eval/results/`.
- **Visual judge** (`stl_agent/judge/`, `eval/judge.py`): renders each generated part from six views and asks
  a multimodal model whether it matches the request, catching parts with the right size but the wrong shape.
  Calibrated against reference parts and hand-graded labels; renders are kept in `eval/renders/`.

## Roadmap

Done: the generation pipeline with retrieval and intake gate, the evaluation benchmark with RAG ablation,
the visual judge, the Thingiverse fallback, combined parts, a fully local mode (Ollama), and a better repair message
when only a part's size is off. Open work is tracked in
[GitHub issues](../../issues): the final report, a Jev gate backend, a stronger
judge model, whether retrieved examples make the model skip requested features, running the visual judge during
builds (optional), editing a built part instead of rebuilding it, and for combined parts: checking that downloaded
models are the right object, showing each piece, and copies of a piece; and building organic shapes with signed
distance functions instead of downloading them; and fine-tuning a small local model on verified builds.
