"""Gradio front end for the text-to-STL CadQuery agent.

Run:  pixi run -e cap6938 app   (or: python -m stl_agent.app)
Then open http://127.0.0.1:7860 in the browser.

Flow: the user chats with the intake gate (`run_gate`), which asks clarifying questions
or rejects out-of-scope requests. Once it is ready, it shows the spec it will build and the
"Build it" button runs `run_pipeline` on that spec, streaming the agent log live.
Requests the gate rules out of scope (sculpted or organic shapes) get a Thingiverse search instead (#9), when
THINGIVERSE_TOKEN is set: the results show as a gallery, and clicking one downloads it (`run_fetch`).
A "composite" plan (#10: pieces to download or generate, then join) is built by `run_assembly`; with
thingiverse_pick = "manual", Build first shows a gallery per downloaded piece for the user to pick from.
"""

import hashlib
import json
import queue
import re
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import trimesh

from .config import CACHE_DIR, OUTPUT_DIR, defaults, known_providers, load_dotenv, set_setting, settings, unset_setting
from .gate import CLARIFY, OUT_OF_SCOPE
from .llm import PROVIDERS, list_models, make_llm
from .pipeline import describe_settings, reset_backends, run_assembly, run_fetch, run_gate, run_pipeline
from .sources import thingiverse

OUTPUT_DIR.mkdir(exist_ok=True)

EXAMPLES = [
    "A 60mm x 40mm plate, 5mm thick, with four 4mm holes 5mm in from each corner",
    "An 80mm diameter flange, 6mm thick, with six 5mm holes on a 60mm bolt circle",
    "A stand for my phone",
    "A hollow open-top box, 50x30x20mm, with 2mm walls",
    "A statue of a horse",
]


def cache_key(request: str) -> str:
    """Stable across runs (unlike hash()); use the same function when saving demo results."""
    return hashlib.md5(request.strip().lower().encode()).hexdigest()[:12]


def load_cached(request: str):
    """Offline fallback: demo_cache/<cache_key>/ holding model.stl, code.py and report.json."""
    folder = CACHE_DIR / cache_key(request)
    if (folder / "model.stl").exists():
        return (
            str(folder / "model.stl"),
            (folder / "code.py").read_text(),
            json.loads((folder / "report.json").read_text()),
        )
    return None


PREVIEW_COLOR = (70, 130, 200, 255)  # RGBA; STL has no color of its own


def make_preview(stl_path: str) -> str:
    """Colored GLB copy of the STL for the 3D viewer (the STL stays the download).

    CadQuery is Z-up and glTF is Y-up, so rotate the part to stand upright in the viewer.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.visual.face_colors = PREVIEW_COLOR
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb = Path(stl_path).with_suffix(".preview.glb")
    mesh.export(glb)
    return str(glb)


def new_state() -> dict:
    # conversation: turns the gate sees; rounds: clarifying rounds asked;
    # spec / name: ready-to-build spec and its file-name slug
    # candidates: Thingiverse results shown in the gallery (sources.thingiverse.Candidate)
    # pieces: a composite plan (#10); picks: piece id -> chosen candidate (None: build its fallback);
    # picking: the piece whose results the gallery shows; assemble: all picked, ready to build
    return {"conversation": [], "rounds": 0, "spec": None, "name": "", "example_ids": None, "candidates": [],
            "pieces": [], "picks": {}, "picking": None, "assemble": False}


def gallery_update(candidates: list | None = None):
    """The Thingiverse results gallery: shown with these candidates, or hidden when there are none."""
    if not candidates:
        return gr.Gallery(value=None, visible=False)
    return gr.Gallery(value=[(c.thumbnail, c.caption()) for c in candidates], visible=True)


def fetch_model(c, state: dict, log) -> str:
    """Download one Thingiverse model as the result (run_fetch). Returns the chat message; raises on failure."""
    stl, report = run_fetch(c, log, name=state.get("name") or None)
    others = report["source"]["other_stl_files"]
    msg = (f"Downloaded [{c.name}]({c.url}) by {c.author} ({c.license}). Saved as `{Path(stl).name}`"
           + ("." if report.get("success") else "; the mesh has issues (see the validation report)."))
    if others:
        shown = ", ".join(f"`{n}`" for n in others[:5]) + (" and more" if len(others) > 5 else "")
        msg += f"\n\nThis took the largest STL; the thing also has {shown} on its page."
    state["conversation"].append({"role": "assistant", "content": f"Downloaded '{c.name}' from Thingiverse."})
    attribution = f"# Downloaded from Thingiverse, not generated: no CadQuery code.\n# {c.credit()}\n"
    state["result"] = {"stl": stl, "code": attribution, "report": report}
    return msg


def search_thingiverse(d, state: dict) -> str:
    """Reply for an out-of-scope request: Thingiverse results (stored in state) or the reason there are none."""
    reason = d.scope_reason or "it needs sculpted or organic geometry."
    if not thingiverse.has_token():  # Thingiverse is optional: decline as before
        return format_decision(d)
    try:
        state["candidates"] = thingiverse.search(d.search)
    except thingiverse.ThingiverseError as e:
        return format_decision(d) + f"\n\nThingiverse search failed: {e}"
    if not state["candidates"]:
        return format_decision(d) + f"\n\nThingiverse has nothing for \"{d.search}\" either."
    if settings().thingiverse_pick == "auto":  # no gallery: take the first result that downloads
        failed = []
        for c in state["candidates"]:
            try:
                msg = fetch_model(c, state, lambda line: None)
            except Exception as e:  # try the next result
                failed.append(f"'{c.name}' ({e})")
                continue
            state["candidates"] = []
            return (f"I can't build this with parametric CAD: {reason}\n\nUsing the first Thingiverse result for "
                    f"\"{d.search}\" instead (change this under ⚙ Settings). {msg} Check its license before sharing "
                    "or selling prints.")
        reason += f"\n\nNo result downloaded ({'; '.join(failed)}), so pick one yourself."
    return (f"I can't build this with parametric CAD: {reason}\n\n"
            f"Here are existing models from Thingiverse for \"{d.search}\". Click one to download it; "
            "check its license before sharing or selling prints.")


RELATION_WORDS = {"on_top": "on top of", "under": "under", "inside": "inside"}


def describe_piece(p: dict) -> str:
    """'cat (from Thingiverse, 30 mm tall), on top of tree'"""
    how = "from Thingiverse" if p["source"] == "thingiverse" else "built with CadQuery"
    size = f", {p['height_mm']:g} mm tall" if p.get("height_mm") else ""
    text = f"**{p['id']}** ({how}{size})"
    if p.get("place"):
        w = p["place"]
        rel = f"{w['side']} of" if w["relation"] == "beside" else RELATION_WORDS[w["relation"]]
        text += f", {rel} {w['on']}"
    return text + (f": {p['spec']}" if p["source"] == "cadquery" else "")


def format_decision(d) -> str:
    if d.action == OUT_OF_SCOPE:
        return (
            f"I can't build this with parametric CAD: {d.scope_reason or 'it needs sculpted or organic geometry.'}\n\n"
            "Try a mechanical or geometric part, or ask for a simplified geometric version."
        )
    if d.action == CLARIFY:
        qs = "\n".join(f"{i}. {q}" for i, q in enumerate(d.questions, 1))
        return f"A few details before I build it:\n\n{qs}\n\nYou can also say \"you choose\" and I'll pick sensible values."

    parts = [f"Here's what I'll build:\n\n> {d.spec}"]
    if d.pieces:
        parts.append("It's made from these pieces, joined into one part:\n"
                     + "\n".join(f"{i}. {describe_piece(p)}" for i, p in enumerate(d.pieces, 1)))
    if d.scope == "simplified":
        parts.append(f"This will be a simplified geometric version ({d.scope_reason}).")
    if d.assumptions:
        parts.append("Assumptions:\n" + "\n".join(f"- {a.get('param')}: {a.get('default')}" for a in d.assumptions))
    if d.note:
        parts.append(d.note)
    parts.append("Click **Build it**, or reply with changes.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Event handlers
#
# Gradio's 3D viewer stays blank if it is updated from a streaming (generator) handler,
# so the streaming handlers only update the chat and log, put any finished result in
# state["result"], and `show_result` fills the result panels in a separate, plain step.
# ---------------------------------------------------------------------------
def on_message(message: str, chat: list, state: dict, use_cache: bool):
    """Returns (textbox, chat, state, build_button, gallery)."""
    message = (message or "").strip()
    if not message:
        yield "", chat, state, gr.Button(interactive=state["spec"] is not None), gr.skip()
        return

    chat = chat + [{"role": "user", "content": message}]

    if use_cache and not state["conversation"] and (cached := load_cached(message)):
        stl, code, report = cached
        state["result"] = {"stl": stl, "code": code, "report": report}
        chat.append({"role": "assistant", "content": "Loaded a cached result (preview on the right)."})
        yield "", chat, state, gr.Button(interactive=False), gr.skip()
        return

    state["conversation"].append({"role": "user", "content": message})
    state.update(spec=None, candidates=[], pieces=[], picks={}, picking=None, assemble=False)
    yield ("", chat + [{"role": "assistant", "content": "_Checking your request..._"}], state,
           gr.Button(interactive=False), gallery_update())

    try:
        d = run_gate(state["conversation"], state["rounds"])
    except Exception as e:  # e.g. quota exhausted: show it in the chat instead of crashing
        state["conversation"].pop()
        chat.append({"role": "assistant", "content": f"Intake check failed: {e}"})
        yield "", chat, state, gr.Button(interactive=False), gr.skip()
        return

    if d.action == OUT_OF_SCOPE:
        if thingiverse.has_token():
            yield ("", chat + [{"role": "assistant", "content": "_Searching Thingiverse..._"}], state,
                   gr.Button(interactive=False), gr.skip())
        reply = search_thingiverse(d, state)
    else:
        reply = format_decision(d)
    if d.action == CLARIFY:
        state["rounds"] += 1
    elif d.action != OUT_OF_SCOPE:
        state["spec"], state["name"] = d.spec, d.name
        state["example_ids"], state["pieces"] = d.example_ids, d.pieces
    state["conversation"].append({"role": "assistant", "content": reply})
    chat.append({"role": "assistant", "content": reply})
    yield "", chat, state, gr.Button(interactive=state["spec"] is not None), gallery_update(state["candidates"])


def download_button(stl: str | None = None):
    if stl:
        return gr.DownloadButton(label=f"Download {Path(stl).name}", value=stl, interactive=True)
    return gr.DownloadButton(label="Download STL", value=None, interactive=False)


def clear_result():
    """Empty the result panels when a new build starts. Returns (viewer, download, code, report)."""
    return None, download_button(), "", None


def _stream(job, chat: list, state: dict, label: str):
    """Run job(log) -> (stl, code, report) in a thread, streaming the log. Yields (chat, state, build_button, log)."""
    messages: queue.Queue = queue.Queue()
    outcome: dict = {}

    def worker():
        try:
            outcome["value"] = job(messages.put)
        except Exception as e:  # surface failures in the UI instead of crashing
            outcome["error"] = e
        finally:
            messages.put(None)  # sentinel: job finished

    threading.Thread(target=worker, daemon=True).start()

    chat = chat + [{"role": "assistant", "content": "_Building... (see the agent log)_"}]
    log_text = ""
    while (msg := messages.get()) is not None:
        log_text += msg + "\n"
        yield chat, state, gr.Button(interactive=False), log_text

    state["rounds"] = 0  # later edits get a fresh clarification budget
    if "error" in outcome:
        chat[-1] = {"role": "assistant", "content": f"Build failed: {outcome['error']}\n\nReply with changes to try again."}
        yield chat, state, gr.Button(interactive=True), log_text + f"\nFAILED: {outcome['error']}"
        return

    stl, code, report = outcome["value"]
    done = "Done, the preview is on the right." if report.get("success") else "Done with warnings (see the validation report)."
    text = f"{done} Saved as `{Path(stl).name}`."
    if report.get("pieces"):
        text += "\n\nPieces:\n" + "\n".join(
            f"- **{r['id']}**: {r.get('model') or r['source']}" + (" (best effort: it didn't pass every check)"
                                                                   if r.get("warning") else "")
            for r in report["pieces"])
        if report.get("merge_notes"):
            text += "\n\n" + " ".join(report["merge_notes"])
    chat[-1] = {"role": "assistant", "content": text + "\n\nReply with changes to adjust it."}
    state["conversation"].append({"role": "assistant", "content": f"Built the {label} from this spec: {state['spec']}"})
    state.update(spec=None, pieces=[], picks={}, picking=None, candidates=[])
    state["result"] = {"stl": stl, "code": code, "report": report}
    yield chat, state, gr.Button(interactive=False), log_text


def next_pick(state: dict) -> tuple[list[str], bool]:
    """Find results for the next Thingiverse piece the user hasn't picked, trying its search phrases in turn.
    Pieces with no results get None (their fallback is built). Returns (chat lines, gallery shown)."""
    lines = []
    for p in state["pieces"]:
        if p["source"] != "thingiverse" or p["id"] in state["picks"]:
            continue
        for phrase in p["search"]:
            if candidates := thingiverse.search(phrase):
                state["candidates"], state["picking"] = candidates, p["id"]
                lines.append(f"Pick the **{p['id']}** from the Thingiverse results for \"{phrase}\" (below the chat).")
                return lines, True
        state["picks"][p["id"]] = None
        lines.append(f"No Thingiverse model found for the {p['id']}; a simplified one will be built instead.")
    state["candidates"], state["picking"] = [], None
    return lines, False


def _assembly_job(state: dict):
    picks = state["picks"] if settings().thingiverse_pick == "manual" else None
    return lambda log: run_assembly(state["pieces"], log, name=state.get("name"), picks=picks)


def on_build(chat: list, state: dict):
    """Returns (chat, state, build_button, log, gallery)."""
    spec = state.get("spec")
    if not spec:
        yield chat, state, gr.Button(interactive=False), "Nothing to build yet.", gr.skip()
        return

    pieces = state.get("pieces") or []
    if not pieces:
        job = lambda log: run_pipeline(spec, log, name=state.get("name"), example_ids=state.get("example_ids"))
        for out in _stream(job, chat, state, "part"):
            yield *out, gr.skip()
        return

    if any(p["source"] == "thingiverse" for p in pieces) and not thingiverse.has_token():
        chat = chat + [{"role": "assistant", "content": f"Some pieces come from Thingiverse: set "
                        f"{thingiverse.TOKEN_VAR} in `.env` to build this, or ask for a fully geometric version."}]
        yield chat, state, gr.Button(interactive=True), "", gr.skip()
        return
    if settings().thingiverse_pick == "manual":
        state["picks"] = {}
        yield chat, state, gr.Button(interactive=False), "Searching Thingiverse...", gr.skip()
        try:
            lines, showing = next_pick(state)
        except thingiverse.ThingiverseError as e:
            chat = chat + [{"role": "assistant", "content": f"Thingiverse search failed: {e}"}]
            yield chat, state, gr.Button(interactive=True), "", gr.skip()
            return
        if lines:
            chat = chat + [{"role": "assistant", "content": "\n\n".join(lines)}]
        if showing:
            yield chat, state, gr.Button(interactive=False), "", gallery_update(state["candidates"])
            return
    for out in _stream(_assembly_job(state), chat, state, "combined part"):
        yield *out, gallery_update()


def on_assemble(chat: list, state: dict):
    """After the last manual pick: build the combined part. Returns (chat, state, build_button, log)."""
    if not state.get("assemble"):
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    state["assemble"] = False
    yield from _stream(_assembly_job(state), chat, state, "combined part")


def pick_piece(c, chat: list, state: dict):
    """A gallery click while picking models for a combined part. Returns (chat, state, log, gallery)."""
    piece = state["picking"]
    state["picks"][piece] = c
    note = "" if c.allows_derivatives else " Its license doesn't allow derivatives, so don't share the combined part."
    lines = [f"Using [{c.name}]({c.url}) by {c.author} for the {piece}.{note}"]
    try:
        more, showing = next_pick(state)
    except thingiverse.ThingiverseError as e:
        state["picking"] = None
        chat = chat + [{"role": "assistant", "content": "\n\n".join(lines + [f"Thingiverse search failed: {e}"])}]
        return chat, state, "", gallery_update()
    chat = chat + [{"role": "assistant", "content": "\n\n".join(lines + more)}]
    if showing:
        return chat, state, gr.skip(), gallery_update(state["candidates"])
    state["assemble"] = True  # on_assemble, next in the chain, builds it
    return chat, state, "", gallery_update()


def on_pick(evt: gr.SelectData, chat: list, state: dict):
    """Download the Thingiverse result clicked in the gallery, or, while picking pieces for a combined part,
    take it for the current piece. Returns (chat, state, log, gallery)."""
    candidates = state.get("candidates") or []
    if not isinstance(evt.index, int) or not 0 <= evt.index < len(candidates):
        return chat, state, gr.skip(), gr.skip()
    c = candidates[evt.index]
    if state.get("picking"):
        return pick_piece(c, chat, state)
    lines: list[str] = []
    try:
        msg = fetch_model(c, state, lines.append)
    except Exception as e:  # network or file errors: show them in the chat
        chat = chat + [{"role": "assistant", "content": f"Download of '{c.name}' failed: {e}"}]
        return chat, state, "\n".join(lines + [f"FAILED: {e}"]), gr.skip()
    chat = chat + [{"role": "assistant", "content": msg}]
    state["candidates"] = []  # done: hide the gallery, so it doesn't look like it wants another pick
    return chat, state, "\n".join(lines), gallery_update()


def show_result(state: dict):
    """Fill the result panels from a finished build. Returns (state, viewer, download, code, report)."""
    result = state.pop("result", None)
    if not result:
        return state, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    stl = result["stl"]
    try:
        preview = make_preview(stl)
    except Exception:  # the viewer can show the raw STL if conversion fails
        preview = stl
    return state, preview, download_button(stl), result["code"], result["report"]


def on_example(evt: gr.SelectData):
    """Clicking an example in the empty chat puts its text in the message box."""
    return evt.value["text"]


def on_reset():
    return [], new_state(), gr.Button(interactive=False), "", None, download_button(), "", None, gallery_update()


# ---------------------------------------------------------------------------
# Model settings panel: provider, model, gate model and address, saved to the settings file.
# Fields hold the current value; an empty field (or the default value) leaves the setting at its default.
# ---------------------------------------------------------------------------
FORM_KEYS = ("model", "gate_model", "base_url")


def key_status(provider: str) -> str:
    load_dotenv()  # pick up a key added to .env while the app runs
    key_env = PROVIDERS[provider]["key_env"]
    if not key_env:
        return "No API key needed (local server)."
    import os

    if os.environ.get(key_env):
        return f"`{key_env}` is set."
    return f"**`{key_env}` is not set.** Add it to `.env` in the project folder; no restart needed."


# Model names with these words are not chat models (speech, images, embeddings, ...); left out of the dropdown.
NON_CHAT = ("embed", "tts", "image", "audio", "live", "transcribe", "translate", "lyria", "aqa", "banana",
            "imagen", "veo", "research", "computer-use", "robotics", "antigravity")


def chat_models(models: list[str]) -> list[str]:
    return [m for m in models if not any(word in m.lower() for word in NON_CHAT)]


def has_key(provider: str) -> bool:
    import os

    key_env = PROVIDERS[provider]["key_env"]
    return not key_env or bool(os.environ.get(key_env))


SAME_AS_MODEL = ("Same as the model", "")


def gate_dropdown(models: list[str], value: str | None = None):
    """The gate model's choices: the build model (empty value, the default) or any of the server's models.
    `value` None leaves the current choice."""
    kw = {} if value is None else {"value": value}
    return gr.Dropdown(choices=[SAME_AS_MODEL, *models], **kw)


def provider_form(provider: str):
    """Fields for one provider from the current settings, with the server's models in the dropdown.
    Returns (model, gate_model, base_url, status)."""
    provider = provider.lower()
    current, default = settings().providers[provider], defaults()["providers"][provider]
    model = current["model"]
    status, choices = key_status(provider), [model]
    if has_key(provider):
        try:
            choices = sorted(set(chat_models(list_models(provider, timeout=3))) | {model})
        except Exception as e:
            status += f" Couldn't list the server's models: {e}"
    url_default = default.get("base_url") or "the provider's own"
    return (
        gr.Dropdown(choices=choices, value=model),
        gate_dropdown(choices, current.get("gate_model") or ""),
        gr.Textbox(value=current.get("base_url") or "", placeholder=f"default: {url_default}"),
        status,
    )


def load_form():
    """Fill the panel when the page opens (the settings file may have changed since the app started)."""
    return settings().provider, *provider_form(settings().provider), describe_settings()


def save_settings(provider: str, model: str, gate_model: str, base_url: str) -> str:
    """Write the panel to the settings file. The next request uses it (cached backends are dropped)."""
    provider = (provider or "").lower()
    default = defaults()["providers"].get(provider)
    if default is None:
        return f"Not saved: unknown provider {provider!r}."
    values = dict(zip(FORM_KEYS, ((v or "").strip() for v in (model, gate_model, base_url))))
    try:
        changes = [("provider", provider, defaults()["provider"])]
        changes += [(f"providers.{provider}.{k}", values[k], default.get(k)) for k in FORM_KEYS]
        for name, value, default_value in changes:
            if not value or value == default_value:
                unset_setting(name)
            else:
                set_setting(name, value)
    except ValueError as e:
        return f"Not saved: {e}"
    reset_backends()
    return "Saved. " + describe_settings()


# Thingiverse section of the panel: who picks downloaded models, for a single out-of-scope part (#9) and for
# each downloaded piece of a combined part (#10).
# It saves as soon as it changes (nothing to test first, unlike the model settings).
PICK_CHOICES = [("I pick each model from a gallery", "manual"), ("Use the first suitable result", "auto")]


def thingiverse_status() -> str:
    load_dotenv()  # pick up a token added to .env while the app runs
    if thingiverse.has_token():
        return f"`{thingiverse.TOKEN_VAR}` is set."
    return (f"**`{thingiverse.TOKEN_VAR}` is not set**, so Thingiverse is off: sculpted parts are declined. Add it to `.env` "
            "(an App Token from thingiverse.com/apps/create); no restart needed.")


def load_thingiverse():
    """Returns (pick mode, token status) for the panel."""
    return settings().thingiverse_pick, thingiverse_status()


def save_pick(mode: str) -> str:
    """Save the pick mode; the default value stays commented out in the settings file."""
    if mode not in dict((v, k) for k, v in PICK_CHOICES):
        return f"Not saved: unknown choice {mode!r}."
    try:
        if mode == defaults()["thingiverse_pick"]:
            unset_setting("thingiverse_pick")
        else:
            set_setting("thingiverse_pick", mode)
    except ValueError as e:
        return f"Not saved: {e}"
    return "Saved. " + thingiverse_status()


def _installed(model: str, models: list[str]) -> bool:
    return model in models or f"{model}:latest" in models  # Ollama lists "name" as "name:latest"


TEST_TIMEOUT = 90  # seconds; a local model may first have to load into memory


def test_connection(provider: str, model: str, base_url: str):
    """List the server's models and send one tiny request with the panel's values (not saved).
    Yields (status, model dropdown, gate model dropdown); the dropdowns get the server's models as choices."""
    provider, model, base_url = (provider or "").lower(), (model or "").strip(), (base_url or "").strip() or None
    yield f"_Testing `{model or 'the default model'}` on {provider}..._", gr.skip(), gr.skip()
    load_dotenv()
    if not has_key(provider):
        yield key_status(provider), gr.skip(), gr.skip()
        return
    lines, dropdowns, listed = [], (gr.skip(), gr.skip()), False
    try:
        models = list_models(provider, base_url)
        listed = True
        choices = sorted(set(chat_models(models)) | ({model} if model else set()))
        dropdowns = gr.Dropdown(choices=choices), gate_dropdown(choices)
        if model and not _installed(model, models):
            hint = f" Install it with `ollama pull {model}`." if provider == "ollama" else ""
            lines.append(f"**`{model}` is not one of the server's {len(models)} models.**{hint}")
    except Exception as e:
        lines.append(f"Couldn't list the server's models: {e}")
    try:
        llm = make_llm(provider, model or None, base_url, timeout=TEST_TIMEOUT, quick=True)
        t0 = time.time()
        reply = llm.complete("You are a connection test. Reply with one word.", [{"role": "user", "content": "Say OK."}])
        answer = f"answered {reply.strip()[:80]!r}" if reply.strip() else "sent an empty reply"
        lines.insert(0, f"**Connected.** `{llm.model}` {answer} ({time.time() - t0:.1f} s for the reply).")
    except Exception as e:
        if "timed out" in str(e):
            reached = " The server answered the model list, so the address and key work." if listed else ""
            lines.append(f"**No reply within {TEST_TIMEOUT} s.**{reached} The model may be busy or still loading: "
                         "try again, or pick another model.")
        else:
            # Providers wrap the reason in JSON; show just its message when there is one.
            m = re.search(r'"message":\s*"([^"]+)"', str(e))
            lines.append(f"**Request failed:** {m.group(1) if m else e}")
    yield "\n\n".join(lines), *dropdowns


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
RESULT_HEIGHT = 280  # px; code and report scroll inside this instead of stretching the page

# The two main columns are equal height; the agent log takes whatever height is left in the
# left column, so its bottom lines up with the code/report box on the right.
CSS = f"""
#code-box .cm-editor {{ height: {RESULT_HEIGHT}px; }}
#settings-cog, #settings-cog:hover {{ font-size: 1.4em; align-self: flex-start; background: transparent;
  border: none; box-shadow: none; }}  /* just the icon, no button background */
#settings-cog:hover {{ opacity: 0.7; }}
#code-box {{ min-height: {RESULT_HEIGHT + 27}px; }}  /* same size while empty; 27px is the label */
#left-col {{ display: flex; flex-direction: column; }}
#log-box {{ flex-grow: 1; display: flex; flex-direction: column; }}
#log-box > label, #log-box .input-container {{ flex-grow: 1; display: flex; flex-direction: column; }}
#log-box textarea {{ flex-grow: 1; height: 100% !important; min-height: 80px; resize: none; }}
"""

# Gradio's own autoscroll doesn't fire once the log's height is set by CSS, so keep it
# scrolled to the newest line whenever its text changes.
HEAD = """
<script>
setInterval(() => {
  const t = document.querySelector('#log-box textarea');
  if (t && t.dataset.len !== String(t.value.length)) {
    t.dataset.len = t.value.length;
    t.scrollTop = t.scrollHeight;
  }
}, 250);
</script>
"""
LAUNCH_KW = dict(css=CSS, head=HEAD)

with gr.Blocks(title="Text-to-STL Agent") as demo:
    with gr.Row(equal_height=True):
        gr.Markdown("# Text-to-STL CAD Agent\nDescribe a part. The agent asks about missing details, then generates "
                    "CadQuery code and an STL.")
        cog = gr.Button("⚙", scale=0, min_width=48, elem_id="settings-cog")
    state = gr.State(new_state())

    # Settings live in a side panel, closed at first; the cog or the panel's edge arrow opens and closes it.
    with gr.Sidebar(label="Settings", open=False, position="right", width=360) as settings_panel:
        gr.Markdown("## Model settings\nWhich model builds the parts. Saved to the settings file; the next "
                    "request uses it, no restart needed. API keys go in `.env`.")
        provider_in = gr.Dropdown(sorted(known_providers()), label="Provider")
        model_in = gr.Dropdown(label="Model", allow_custom_value=True)
        gate_in = gr.Dropdown(label="Gate model (intake questions)", allow_custom_value=True)
        url_in = gr.Textbox(label="Server address")
        key_md = gr.Markdown()
        with gr.Row():
            test_btn = gr.Button("Test connection")
            save_btn = gr.Button("Save", variant="primary")
        settings_md = gr.Markdown()

        gr.Markdown("## Thingiverse\nWho chooses downloaded models, for a sculpted part (a horse statue) and for "
                    "each piece of a joined part (a cat on a tree). Saved when you change it.")
        pick_in = gr.Radio(PICK_CHOICES, label="Picking models")
        pick_md = gr.Markdown()

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="left-col"):
            chat = gr.Chatbot(label="Chat", height=340, examples=[{"text": e} for e in EXAMPLES])
            # Thingiverse results for out-of-scope requests; hidden until there are some.
            gallery = gr.Gallery(label="Thingiverse models (click one to download it)", visible=False,
                                 columns=3, height=260, allow_preview=False)
            msg = gr.Textbox(show_label=False, placeholder="e.g. A 60x40x5mm plate with four 4mm holes...", lines=2)
            with gr.Row():
                send = gr.Button("Send")
                build = gr.Button("Build it", variant="primary", interactive=False)
                reset = gr.Button("New part")
            use_cache = gr.Checkbox(label="Use cached demo results if available (offline fallback)", value=False)
            log_box = gr.Textbox(label="Agent log", lines=4, max_lines=4, interactive=False, elem_id="log-box")

        with gr.Column(scale=1):
            viewer = gr.Model3D(label="Preview", clear_color=(0.95, 0.95, 0.95, 1.0), height=280)
            download = download_button()
            with gr.Tabs():
                with gr.Tab("Generated code"):
                    code_box = gr.Code(language="python", label="CadQuery", elem_id="code-box")
                with gr.Tab("Validation report"):
                    report_box = gr.JSON(height=RESULT_HEIGHT, max_height=RESULT_HEIGHT)


    results = [viewer, download, code_box, report_box]
    for trigger in (send.click, msg.submit):
        trigger(on_message, inputs=[msg, chat, state, use_cache], outputs=[msg, chat, state, build, gallery]).then(
            show_result, inputs=state, outputs=[state, *results]
        )
    build.click(clear_result, outputs=results).then(
        on_build, inputs=[chat, state], outputs=[chat, state, build, log_box, gallery]
    ).then(show_result, inputs=state, outputs=[state, *results])
    # on_pick goes first: only the first handler of a chain gets the click (gr.SelectData).
    gallery.select(on_pick, inputs=[chat, state], outputs=[chat, state, log_box, gallery]).then(
        on_assemble, inputs=[chat, state], outputs=[chat, state, build, log_box]  # after the last pick (#10)
    ).then(show_result, inputs=state, outputs=[state, *results])
    chat.example_select(on_example, outputs=msg)
    reset.click(on_reset, outputs=[chat, state, build, log_box, viewer, download, code_box, report_box, gallery])

    form = [model_in, gate_in, url_in, key_md]
    # Gradio's own progress overlay would cover the fields and show a countdown guessed from the last run;
    # test_connection writes its own "Testing..." line instead.
    quiet = dict(show_progress="hidden")
    # The cog toggles the panel; the panel's own arrow works too, so track its state from its events.
    panel_open = gr.State(False)
    settings_panel.expand(lambda: True, outputs=panel_open, **quiet)
    settings_panel.collapse(lambda: False, outputs=panel_open, **quiet)
    cog.click(lambda is_open: (gr.Sidebar(open=not is_open), not is_open), inputs=panel_open,
              outputs=[settings_panel, panel_open], **quiet)
    demo.load(load_form, outputs=[provider_in, *form, settings_md], **quiet)
    provider_in.input(provider_form, inputs=provider_in, outputs=form, **quiet)
    test_btn.click(test_connection, inputs=[provider_in, model_in, url_in], outputs=[settings_md, model_in, gate_in], **quiet)
    save_btn.click(save_settings, inputs=[provider_in, model_in, gate_in, url_in], outputs=settings_md, **quiet)
    demo.load(load_thingiverse, outputs=[pick_in, pick_md], **quiet)
    pick_in.input(save_pick, inputs=pick_in, outputs=pick_md, **quiet)

if __name__ == "__main__":
    print(describe_settings())
    demo.launch(**LAUNCH_KW)
