"""Gradio front end for the text-to-STL CadQuery agent.

Run:  pixi run -e cap6938 app
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

from agent import run_gate, run_pipeline
from gate import CLARIFY, OUT_OF_SCOPE

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cached demo results: request text -> folder in demo_cache/ containing
# model.stl, code.py and report.json. Used as an offline fallback.
CACHE_DIR = Path("demo_cache")

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
    folder = CACHE_DIR / cache_key(request)
    if (folder / "model.stl").exists():
        return (
            str(folder / "model.stl"),
            (folder / "code.py").read_text(),
            json.loads((folder / "report.json").read_text()),
        )
    return None


def new_state() -> dict:
    # conversation: turns the gate sees; rounds: clarifying rounds asked; spec: ready-to-build spec
    return {"conversation": [], "rounds": 0, "spec": None}


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
# ---------------------------------------------------------------------------
def on_message(message: str, chat: list, state: dict, use_cache: bool):
    """Returns (textbox, chat, state, build_button, viewer, download, code, report)."""
    message = (message or "").strip()
    keep = gr.skip()
    if not message:
        yield "", chat, state, gr.Button(interactive=state["spec"] is not None), keep, keep, keep, keep
        return

    chat = chat + [{"role": "user", "content": message}]

    if use_cache and not state["conversation"] and (cached := load_cached(message)):
        stl, code, report = cached
        chat.append({"role": "assistant", "content": "Loaded a cached result (preview on the right)."})
        yield "", chat, state, gr.Button(interactive=False), stl, stl, code, report
        return

    state["conversation"].append({"role": "user", "content": message})
    state["spec"] = None
    yield "", chat + [{"role": "assistant", "content": "_Checking your request..._"}], state, gr.Button(interactive=False), keep, keep, keep, keep

    try:
        d = run_gate(state["conversation"], state["rounds"])
    except Exception as e:  # e.g. quota exhausted: show it in the chat instead of crashing
        state["conversation"].pop()
        chat.append({"role": "assistant", "content": f"Intake check failed: {e}"})
        yield "", chat, state, gr.Button(interactive=False), keep, keep, keep, keep
        return

    reply = format_decision(d)
    if d.action == CLARIFY:
        state["rounds"] += 1
    elif d.action != OUT_OF_SCOPE:
        state["spec"] = d.spec
    state["conversation"].append({"role": "assistant", "content": reply})
    chat.append({"role": "assistant", "content": reply})
    yield "", chat, state, gr.Button(interactive=state["spec"] is not None), keep, keep, keep, keep


def on_build(chat: list, state: dict):
    """Returns (chat, state, build_button, log, viewer, download, code, report)."""
    spec = state.get("spec")
    if not spec:
        yield chat, state, gr.Button(interactive=False), "Nothing to build yet.", None, None, "", None
        return

    messages: queue.Queue = queue.Queue()
    outcome: dict = {}

    def worker():
        try:
            outcome["value"] = run_pipeline(spec, messages.put)
        except Exception as e:  # surface failures in the UI instead of crashing
            outcome["error"] = e
        finally:
            messages.put(None)  # sentinel: pipeline finished

    threading.Thread(target=worker, daemon=True).start()

    chat = chat + [{"role": "assistant", "content": "_Building... (see the agent log)_"}]
    log_text = ""
    while (msg := messages.get()) is not None:
        log_text += msg + "\n"
        yield chat, state, gr.Button(interactive=False), log_text, None, None, "", None

    state["rounds"] = 0  # later edits get a fresh clarification budget
    if "error" in outcome:
        chat[-1] = {"role": "assistant", "content": f"Build failed: {outcome['error']}\n\nReply with changes to try again."}
        yield chat, state, gr.Button(interactive=True), log_text + f"\nFAILED: {outcome['error']}", None, None, "", None
        return

    stl, code, report = outcome["value"]
    done = "Done, the preview is on the right." if report.get("success") else "Done with warnings (see the validation report)."
    chat[-1] = {"role": "assistant", "content": done + " Reply with changes to adjust it."}
    state["conversation"].append({"role": "assistant", "content": f"Built the part from this spec: {spec}"})
    state["spec"] = None
    yield chat, state, gr.Button(interactive=False), log_text, stl, stl, code, report


def on_reset():
    return [], new_state(), gr.Button(interactive=False), "", None, None, "", None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Text-to-STL Agent") as demo:
    gr.Markdown("# Text-to-STL CAD Agent\nDescribe a part. The agent asks about missing details, then generates CadQuery code and an STL.")
    state = gr.State(new_state())

    with gr.Row():
        with gr.Column(scale=1):
            chat = gr.Chatbot(label="Chat", height=420)
            msg = gr.Textbox(show_label=False, placeholder="e.g. A 60x40x5mm plate with four 4mm holes...", lines=2)
            with gr.Row():
                send = gr.Button("Send")
                build = gr.Button("Build it", variant="primary", interactive=False)
                reset = gr.Button("New part")
            use_cache = gr.Checkbox(label="Use cached demo results if available (offline fallback)", value=False)
            gr.Examples(EXAMPLES, inputs=msg)
            with gr.Accordion("Agent log", open=False):
                log_box = gr.Textbox(show_label=False, lines=12, interactive=False)

        with gr.Column(scale=1):
            viewer = gr.Model3D(label="Preview", clear_color=(0.95, 0.95, 0.95, 1.0))
            download = gr.File(label="Download STL")
            with gr.Tabs():
                with gr.Tab("Generated code"):
                    code_box = gr.Code(language="python", label="CadQuery")
                with gr.Tab("Validation report"):
                    report_box = gr.JSON()

    msg_io = dict(
        fn=on_message,
        inputs=[msg, chat, state, use_cache],
        outputs=[msg, chat, state, build, viewer, download, code_box, report_box],
    )
    send.click(**msg_io)
    msg.submit(**msg_io)
    build.click(
        on_build,
        inputs=[chat, state],
        outputs=[chat, state, build, log_box, viewer, download, code_box, report_box],
    )
    reset.click(on_reset, outputs=[chat, state, build, log_box, viewer, download, code_box, report_box])

if __name__ == "__main__":
    demo.launch()
