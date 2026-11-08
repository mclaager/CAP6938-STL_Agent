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
With organic_builder = "sdf" (#42), organic objects and pieces are sculpted with SDF code instead
(`pipeline.sculpt_organic`); a failed sculpted part, or one that isn't one closed solid, falls back to the
Thingiverse search.
The settings panel (⚙) is in settings_panel.py.
"""

import queue
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import trimesh

from .config import OUTPUT_DIR, settings
from .gate import CLARIFY, OUT_OF_SCOPE
from .pipeline import describe_settings, is_one_solid, run_assembly, run_fetch, run_gate, run_pipeline
from .settings_panel import build_panel
from .sources import thingiverse

OUTPUT_DIR.mkdir(exist_ok=True)

EXAMPLES = [
    "A 60mm x 40mm plate, 5mm thick, with four 4mm holes 5mm in from each corner",
    "An 80mm diameter flange, 6mm thick, with six 5mm holes on a 60mm bolt circle",
    "A stand for my phone",
    "A hollow open-top box, 50x30x20mm, with 2mm walls",
    "A statue of a horse",
]


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
    # language: "sdf" when the spec is an organic object to sculpt (#42); search: its Thingiverse search words,
    # used if sculpting fails
    return {"conversation": [], "rounds": 0, "spec": None, "name": "", "example_ids": None, "candidates": [],
            "pieces": [], "picks": {}, "picking": None, "assemble": False, "language": "cadquery", "search": ""}


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
    auto = settings().thingiverse_pick == "auto"
    words = thingiverse.object_words(d.name, d.search)
    try:
        # Auto: only results named or tagged as the object (#39). The gallery shows those first.
        state["candidates"] = thingiverse.search(d.search, words=words, strict=auto)
        if auto and not state["candidates"]:  # nothing matched: let the user pick from everything found
            auto = False
            state["candidates"] = thingiverse.search(d.search, words=words)
            reason += (f"\n\nNo result is named or tagged \"{'/'.join(sorted(words))}\", so pick one yourself"
                       " (or none).")
    except thingiverse.ThingiverseError as e:
        return format_decision(d) + f"\n\nThingiverse search failed: {e}"
    if not state["candidates"]:
        return format_decision(d) + f"\n\nThingiverse has nothing for \"{d.search}\" either."
    if auto:  # no gallery: take the first result that downloads
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
    how = {"thingiverse": "from Thingiverse", "sdf": "sculpted"}.get(p["source"], "built with CadQuery")
    size = f", {p['height_mm']:g} mm tall" if p.get("height_mm") else ""
    text = f"**{p['id']}** ({how}{size})"
    if p.get("place"):
        w = p["place"]
        rel = f"{w['side']} of" if w["relation"] == "beside" else RELATION_WORDS[w["relation"]]
        text += f", {rel} {w['on']}"
    return text + (f": {p['spec']}" if p["source"] in ("cadquery", "sdf") else "")


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
    if d.scope == "organic":
        parts.append("This will be sculpted from smooth rounded shapes (experimental; change this under ⚙ Settings), "
                     "so small details are simplified.")
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
def on_message(message: str, chat: list, state: dict):
    """Returns (textbox, chat, state, build_button, gallery)."""
    message = (message or "").strip()
    if not message:
        yield "", chat, state, gr.Button(interactive=state["spec"] is not None), gr.skip()
        return

    chat = chat + [{"role": "user", "content": message}]

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
        state["language"], state["search"] = ("sdf" if d.scope == "organic" else "cadquery"), d.search
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
        # The first phrase with a result named or tagged as the piece (#39, listed first); if none has one,
        # the first phrase with any results.
        words, first = thingiverse.object_words(p["id"], p["search"]), None
        for phrase in p["search"]:
            candidates = thingiverse.search(phrase, words=words)
            first = first or (candidates and (phrase, candidates))
            if candidates and candidates[0].match:
                first = (phrase, candidates)
                break
        if first:
            phrase, state["candidates"] = first
            state["picking"] = p["id"]
            lines.append(f"Pick the **{p['id']}** from the Thingiverse results for \"{phrase}\" (below the chat).")
            return lines, True
        state["picks"][p["id"]] = None
        lines.append(f"No Thingiverse model found for the {p['id']}; a simplified one will be built instead.")
    state["candidates"], state["picking"] = [], None
    return lines, False


def _thingiverse_instead(chat: list, state: dict, flawed: bool = False):
    """After a failed sculpted build, or one that isn't a closed solid (`flawed`, its result stays shown): search
    Thingiverse for the object. Returns (chat, state, build_button, log, gallery)."""
    from .gate import OUT_OF_SCOPE, Decision

    reason = ("the sculpted part didn't come out as one closed solid (it's shown, but may not print well)."
              if flawed else "sculpting it didn't work.")
    d = Decision(action=OUT_OF_SCOPE, spec=state["spec"], scope="out_of_scope",
                 scope_reason=reason, search=state.get("search") or state.get("name", ""))
    reply = search_thingiverse(d, state)
    state["conversation"].append({"role": "assistant", "content": reply})
    chat = chat + [{"role": "assistant", "content": reply}]
    log = ("Sculpted part is not one closed solid" if flawed else "Sculpting failed") + "; searched Thingiverse instead."
    return chat, state, gr.Button(interactive=state.get("spec") is not None), log, gallery_update(state["candidates"])


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
        language = state.get("language", "cadquery")
        job = lambda log: run_pipeline(spec, log, name=state.get("name"), example_ids=state.get("example_ids"),
                                       language=language)
        out = None
        for out in _stream(job, chat, state, "part"):
            yield *out, gr.skip()
        if language == "sdf" and thingiverse.has_token():
            # Sculpting failed or left a broken mesh: offer existing models instead, as for any organic object.
            result = state.get("result")
            if not result:
                yield _thingiverse_instead(out[0], state)
            elif not is_one_solid(result["report"].get("validation")):
                state["spec"] = spec  # _stream cleared it; the search needs it
                yield _thingiverse_instead(out[0], state, flawed=True)
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
    code = gr.Code(value=result["code"], label=code_label(result["report"]))
    return state, preview, download_button(stl), code, result["report"]


def code_label(report: dict) -> str:
    """The code panel's title: what kind of code the result came from."""
    if "source" in report:  # run_fetch
        return "Source (downloaded, no code)"
    if report.get("pieces"):
        return "Code for each piece"
    return "SDF script (sculpted)" if report.get("language") == "sdf" else "CadQuery"


def on_example(evt: gr.SelectData):
    """Clicking an example in the empty chat puts its text in the message box."""
    return evt.value["text"]


def on_reset():
    return [], new_state(), gr.Button(interactive=False), "", None, download_button(), "", None, gallery_update()


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
        gr.Markdown("# Text-to-STL CAD Agent\nDescribe a part. The agent asks about missing details, then writes "
                    "code for it (CadQuery, or an SDF script for sculpted shapes) and builds an STL.")
        cog = gr.Button("⚙", scale=0, min_width=48, elem_id="settings-cog")
    state = gr.State(new_state())

    # Settings live in a side panel (settings_panel.py), opened with the cog.
    build_panel(demo, cog)

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
            log_box = gr.Textbox(label="Agent log", lines=4, max_lines=4, interactive=False, elem_id="log-box")

        with gr.Column(scale=1):
            viewer = gr.Model3D(label="Preview", clear_color=(0.95, 0.95, 0.95, 1.0), height=280)
            download = download_button()
            with gr.Tabs():
                with gr.Tab("Generated code"):
                    code_box = gr.Code(language="python", label="Code", elem_id="code-box")
                with gr.Tab("Validation report"):
                    report_box = gr.JSON(height=RESULT_HEIGHT, max_height=RESULT_HEIGHT)


    results = [viewer, download, code_box, report_box]
    for trigger in (send.click, msg.submit):
        trigger(on_message, inputs=[msg, chat, state], outputs=[msg, chat, state, build, gallery]).then(
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


if __name__ == "__main__":
    print(describe_settings())
    demo.launch(**LAUNCH_KW)
