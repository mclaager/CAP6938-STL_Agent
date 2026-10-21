"""Gradio front end for the text-to-STL CadQuery agent.

Run:  pip install gradio cadquery trimesh
      python app.py
Then open http://127.0.0.1:7860 in the browser.

The UI only depends on one function, `run_pipeline`, which you replace with
your real agent. It must yield progress messages as it works and finish by
returning (stl_path, code, report). Everything else here is presentation.
"""

import hashlib
import json
import queue
import threading
from pathlib import Path

import gradio as gr

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cached demo results: request text -> folder in outputs/ containing
# model.stl, code.py and report.json. Used as an offline fallback.
CACHE_DIR = Path("demo_cache")

EXAMPLES = [
    "A 60mm x 40mm plate, 5mm thick, with four 4mm holes 5mm in from each corner",
    "An 80mm diameter flange, 6mm thick, with six 5mm holes on a 60mm bolt circle",
    "An L-bracket with two mounting holes on each leg and filleted outer edges",
    "A hollow open-top box, 50x30x20mm, with 2mm walls",
]


from agent import run_pipeline  # noqa: E402  (real agent; see agent.py)


# ---------------------------------------------------------------------------
# UI plumbing (streams agent logs live while the pipeline runs)
# ---------------------------------------------------------------------------
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


def generate(request: str, use_cache: bool):
    request = (request or "").strip()
    if not request:
        yield "Please enter a request.", None, None, "", None
        return

    if use_cache:
        cached = load_cached(request)
        if cached:
            stl, code, report = cached
            yield "Loaded cached result.", stl, stl, code, report
            return

    messages: queue.Queue = queue.Queue()
    outcome: dict = {}

    def worker():
        try:
            outcome["value"] = run_pipeline(request, messages.put)
        except Exception as e:  # surface failures in the UI instead of crashing
            outcome["error"] = e
        finally:
            messages.put(None)  # sentinel: pipeline finished

    threading.Thread(target=worker, daemon=True).start()

    log_text = ""
    while True:
        msg = messages.get()
        if msg is None:
            break
        log_text += msg + "\n"
        yield log_text, None, None, "", None

    if "error" in outcome:
        yield log_text + f"\nFAILED: {outcome['error']}", None, None, "", None
        return

    stl, code, report = outcome["value"]
    yield log_text, stl, stl, code, report


with gr.Blocks(title="Text-to-STL Agent") as demo:
    gr.Markdown("# Text-to-STL CAD Agent\nDescribe a part and the agent generates CadQuery code and an STL.")

    with gr.Row():
        with gr.Column(scale=1):
            request = gr.Textbox(label="Part description", lines=4, placeholder="e.g. A 60x40x5mm plate with four 4mm holes...")
            use_cache = gr.Checkbox(label="Use cached demo results if available (offline fallback)", value=False)
            go = gr.Button("Generate", variant="primary")
            gr.Examples(EXAMPLES, inputs=request)
            log_box = gr.Textbox(label="Agent log", lines=12, interactive=False)

        with gr.Column(scale=1):
            viewer = gr.Model3D(label="Preview", clear_color=(0.95, 0.95, 0.95, 1.0))
            download = gr.File(label="Download STL")
            with gr.Tabs():
                with gr.Tab("Generated code"):
                    code_box = gr.Code(language="python", label="CadQuery")
                with gr.Tab("Validation report"):
                    report_box = gr.JSON()

    outputs = [log_box, viewer, download, code_box, report_box]
    go.click(generate, inputs=[request, use_cache], outputs=outputs)
    request.submit(generate, inputs=[request, use_cache], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
