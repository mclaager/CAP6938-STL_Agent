# STL Generating Agent

A chat agent that turns a plain-English part description into a 3D-printable STL file. An LLM
writes [CadQuery](https://cadquery.readthedocs.io) code, which runs in a sandbox, is checked as a
mesh, and is repaired in a loop when it fails. Before building, an intake step asks about missing
details that matter and declines shapes CadQuery can't build.

Course project for CAP 6938 (AI Knowledge Systems, Fall 2026).

## Install

Three ways to install it; none needs a GPU.

| | Package versions | Runs on | Best for |
|---|---|---|---|
| [Pixi](#pixi-recommended) (recommended) | exact, from `pixi.lock` | Linux x86-64 | development, benchmark runs |
| [Conda, venv or another environment](#conda-venv-or-another-environment) | minimums only (`requirements.txt`) | Linux (others untested) | an existing Python setup |
| [Docker](#docker) | exact, from `pixi.lock` | x86-64 with Docker or Podman | another machine, reproducing results |

### Pixi (recommended)

Install [pixi](https://pixi.sh), then from the repo root:

```bash
pixi run -e cap6938 app        # installs the environment on the first run; app at http://127.0.0.1:7860
```

Every command is a pixi task in the `cap6938` environment ([Commands](#commands)). The separate `finetune`
environment (PyTorch, CUDA) is only for fine-tuning a local model.

### Conda, venv or another environment

Python 3.12 (the version the lock file uses) and the packages in `requirements.txt`:

```bash
conda create -n stl-agent python=3.12 && conda activate stl-agent    # or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m stl_agent.app                                               # from the repo root
```

`requirements.txt` lists minimum versions only, so pip may install newer packages than the ones tested; for
exact versions use pixi or Docker. Run commands with `python -m` instead of `pixi run`
([Commands](#commands)). On Linux without a desktop, the renders need Mesa's OpenGL libraries (`libegl1`,
`libosmesa6` on Debian/Ubuntu).

### Docker

The image is Ubuntu 24.04 with the pixi environment, and renders with Mesa's software OpenGL. Tested with
Podman on Fedora.

```bash
docker compose up --build        # app at http://127.0.0.1:7860
```

API keys come from `.env` ([Set up](#set-up); optional): compose passes each line to the container as an
environment variable, which is where the app looks for them. `./data/settings.toml` is the container's
settings file and `./outputs` gets the finished parts. Other commands run by their pixi task names
([Commands](#commands)):

```bash
docker compose run --rm app test     # the test suite (no API calls)
docker compose run --rm app bench report --study report-v1 --judge-model gemini-3.8-flash --organic-judge-model claude-sonnet-5-5
```

The second command rebuilds the report's tables from the committed results (`eval/results/report-v1.md`)
without any model requests. Benchmark rows record the git commit; the image has no `.git`, so build with
`GIT_COMMIT=$(git rev-parse --short HEAD) docker compose build` before benchmark runs.

**Local models with a GPU.** `docker compose --profile ollama up --build` also starts Ollama in a container,
with its context window already raised. It uses an NVIDIA GPU through the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (without one,
delete the `deploy` block in `compose.yaml` and it runs on the CPU, slowly). Pull a model once with
`docker compose exec ollama ollama pull gemma4:12b`, then set the server address in the app's settings panel
to `http://ollama:11434/v1`. The app container itself needs no GPU.

Ollama already running on the host instead: set the server address to `http://host.docker.internal:11434/v1`
(Docker; on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]` to the app service) or
`http://host.containers.internal:11434/v1` (Podman), and let Ollama listen beyond localhost
(`OLLAMA_HOST=0.0.0.0` on its service).

Without compose: `docker build -t stl-agent .`, then
`docker run --rm -p 7860:7860 --env-file .env -v ./data:/app/data -v ./outputs:/app/outputs stl-agent`
(create `data` and `outputs` first with Podman; on SELinux add `:z` to each mount).

Software rendering differs from a GPU's in a few edge pixels, so the visual judge can score a part slightly
differently than on the host.

## Set up

API keys go in `.env` in the repo root (Docker reads the same file):

```bash
echo "GEMINI_API_KEY=your-key" > .env      # free key: aistudio.google.com (or ANTHROPIC_API_KEY=... for Claude)
```

The first run creates `settings.toml` in the project root (Docker: `data/settings.toml`), listing every setting with its default, commented
out. To change one, uncomment its line and edit it: for example `provider = "gemini"` for the key above, or
`provider = "ollama"` to run on a local model with [Ollama](https://ollama.com) (no key needed).
Provider, model and server address can also be set in the app, under "Model settings" ("Test connection"
checks them first). `.env` holds only API keys. The `settings` command shows every setting and where its
value came from.

Requests that can't be built with CAD code (statues, animals, characters) get a Thingiverse search instead,
if `.env` has a `THINGIVERSE_TOKEN`: register an app at <https://www.thingiverse.com/apps/create> and copy
its App Token. The results show as thumbnails; clicking one downloads its STL, with the author and license
shown in the chat and saved next to it (`outputs/<name>.source.json`). Thingiverse is optional: without a token, the app
declines sculpted parts, and parts that would join them with others, as it did before.

## Commands

With pixi:

```bash
pixi run -e cap6938 app        # the app
pixi run -e cap6938 test       # tests (no API calls)
pixi run -e cap6938 examples   # rebuild the few-shot example library
pixi run -e cap6938 bench run --budget 18   # benchmark; resumes where it stopped
pixi run -e cap6938 bench report             # results tables
pixi run -e cap6938 judge run --gen-model gemini-3.5-flash-lite   # visual judge over benchmark parts
pixi run -e cap6938 judge sheet              # review page: eval/renders/index.html
pixi run -e cap6938 retrieval-eval           # offline retrieval scores (no API calls)
pixi run -e cap6938 references               # check the benchmark's reference solutions build
pixi run -e cap6938 settings                 # every setting and where its value came from
```

In Docker, the same task names: `docker compose run --rm app bench report`. In another environment, the
module each task runs (listed under `[feature.cap6938.tasks]` in `pixi.toml`): `python -m stl_agent.app`,
`python -m pytest -q`, `python -m scripts.make_examples`, `python -m eval.bench`, `python -m eval.judge`,
`python -m eval.retrieval`, `python -m eval.references`, `python -m stl_agent.config`.

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
6. **Organic shapes** (`stl_agent/sdf.py`): with `organic_builder = "sdf"` (settings panel, "Organic shapes"),
   animals, plants and other organic objects are sculpted with signed distance function code, retrieved from
   `stl_agent/data/sdf_examples.json`, instead of downloaded; a failed sculpt falls back to Thingiverse.
7. **UI** (`stl_agent/app.py`): Gradio chat with a 3D preview, the generated code, and a validation report.

Settings (model, attempts, gate behaviour) are in `settings.toml`; the defaults, with a comment for each, are in
`stl_agent/data/settings.defaults.toml`.

## Evaluation

- **Main suite** (`eval/cases.json`): 48 requests from simple parts to out-of-scope ones, each fixed-size case
  with a hand-written reference solution (`eval/references.py`). A part counts as correct when it is built at
  the expected size (or declined when it should be).
- **Organic suite** (`eval/organic_cases.json`): 30 organic objects (animals, plants, food) with no expected
  size, sculpted with SDF code or downloaded from Thingiverse as a baseline. Scored by the visual judge.
- **Visual judge** (`stl_agent/judge/`, `eval/judge.py`): renders each generated part from six views and asks
  a multimodal model whether it matches the request, catching parts with the right size but the wrong shape.
  Calibrated against reference parts and hand-graded labels; renders are kept in `eval/renders/`.
- **Report round** (`eval/results/report-v1.md`): seven models on the main suite (five also without retrieved
  examples) and four on the organic suite plus the Thingiverse baseline, 3 runs per case; and four judges
  compared against 146 hand-graded parts. The source of every number in the final report; earlier studies are
  in `eval/results/`.

## Roadmap

Open work is tracked in [GitHub issues](https://github.com/mclaager/stl_agent/issues).
