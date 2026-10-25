"""Gradio front end for the text-to-STL CadQuery agent.

Run:  pixi run -e cap6938 app   (or: python -m stl_agent.app)
Then open http://127.0.0.1:7860 in the browser.

Flow: the user chats with the intake gate (`run_gate`), which asks clarifying questions
or rejects out-of-scope requests. Once it is ready, it shows the spec it will build and the
"Build it" button runs `run_pipeline` on that spec, streaming the agent log live.
"""

import hashlib
import json
import queue
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import trimesh

from .config import CACHE_DIR, OUTPUT_DIR
from .gate import CLARIFY, OUT_OF_SCOPE
from .pipeline import describe_settings, run_gate, run_pipeline

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
    return {"conversation": [], "rounds": 0, "spec": None, "name": "", "example_ids": None}


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
    """Returns (textbox, chat, state, build_button)."""
    message = (message or "").strip()
    if not message:
        yield "", chat, state, gr.Button(interactive=state["spec"] is not None)
        return

    chat = chat + [{"role": "user", "content": message}]

    if use_cache and not state["conversation"] and (cached := load_cached(message)):
        stl, code, report = cached
        state["result"] = {"stl": stl, "code": code, "report": report}
        chat.append({"role": "assistant", "content": "Loaded a cached result (preview on the right)."})
        yield "", chat, state, gr.Button(interactive=False)
        return

    state["conversation"].append({"role": "user", "content": message})
    state["spec"] = None
    yield "", chat + [{"role": "assistant", "content": "_Checking your request..._"}], state, gr.Button(interactive=False)

    try:
        d = run_gate(state["conversation"], state["rounds"])
    except Exception as e:  # e.g. quota exhausted: show it in the chat instead of crashing
        state["conversation"].pop()
        chat.append({"role": "assistant", "content": f"Intake check failed: {e}"})
        yield "", chat, state, gr.Button(interactive=False)
        return

    reply = format_decision(d)
    if d.action == CLARIFY:
        state["rounds"] += 1
    elif d.action != OUT_OF_SCOPE:
        state["spec"], state["name"] = d.spec, d.name
        state["example_ids"] = d.example_ids
    state["conversation"].append({"role": "assistant", "content": reply})
    chat.append({"role": "assistant", "content": reply})
    yield "", chat, state, gr.Button(interactive=state["spec"] is not None)


def download_button(stl: str | None = None):
    if stl:
        return gr.DownloadButton(label=f"Download {Path(stl).name}", value=stl, interactive=True)
    return gr.DownloadButton(label="Download STL", value=None, interactive=False)


def clear_result():
    """Empty the result panels when a new build starts. Returns (viewer, download, code, report)."""
    return None, download_button(), "", None


def on_build(chat: list, state: dict):
    """Returns (chat, state, build_button, log)."""
    spec = state.get("spec")
    if not spec:
        yield chat, state, gr.Button(interactive=False), "Nothing to build yet."
        return

    messages: queue.Queue = queue.Queue()
    outcome: dict = {}

    def worker():
        try:
            outcome["value"] = run_pipeline(spec, messages.put, name=state.get("name"),
                                            example_ids=state.get("example_ids"))
        except Exception as e:  # surface failures in the UI instead of crashing
            outcome["error"] = e
        finally:
            messages.put(None)  # sentinel: pipeline finished

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
    chat[-1] = {"role": "assistant", "content": f"{done} Saved as `{Path(stl).name}`. Reply with changes to adjust it."}
    state["conversation"].append({"role": "assistant", "content": f"Built the part from this spec: {spec}"})
    state["spec"] = None
    state["result"] = {"stl": stl, "code": code, "report": report}
    yield chat, state, gr.Button(interactive=False), log_text


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
    return [], new_state(), gr.Button(interactive=False), "", None, download_button(), "", None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
RESULT_HEIGHT = 280  # px; code and report scroll inside this instead of stretching the page

# The two main columns are equal height; the agent log takes whatever height is left in the
# left column, so its bottom lines up with the code/report box on the right.
CSS = f"""
#code-box .cm-editor {{ height: {RESULT_HEIGHT}px; }}
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
    gr.Markdown("# Text-to-STL CAD Agent\nDescribe a part. The agent asks about missing details, then generates CadQuery code and an STL.")
    state = gr.State(new_state())

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="left-col"):
            chat = gr.Chatbot(label="Chat", height=340, examples=[{"text": e} for e in EXAMPLES])
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
        trigger(on_message, inputs=[msg, chat, state, use_cache], outputs=[msg, chat, state, build]).then(
            show_result, inputs=state, outputs=[state, *results]
        )
    build.click(clear_result, outputs=results).then(
        on_build, inputs=[chat, state], outputs=[chat, state, build, log_box]
    ).then(show_result, inputs=state, outputs=[state, *results])
    chat.example_select(on_example, outputs=msg)
    reset.click(on_reset, outputs=[chat, state, build, log_box, viewer, download, code_box, report_box])

if __name__ == "__main__":
    print(describe_settings())
    demo.launch(**LAUNCH_KW)
