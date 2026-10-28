"""The app's settings panel (the ⚙ sidebar): model settings and Thingiverse picking.

Model settings: provider, model, gate model and server address, saved to the settings file. Fields hold the
current value; an empty field (or the default value) leaves the setting at its default. "Test connection"
sends one tiny request with the panel's values without saving them.

`build_panel(demo, cog)` creates the sidebar inside the app's gr.Blocks and wires its events.
"""

import os
import re
import time

import gradio as gr

from .config import defaults, known_providers, load_dotenv, set_setting, settings, unset_setting
from .llm import PROVIDERS, list_models, make_llm
from .pipeline import describe_settings, reset_backends
from .sources import thingiverse

FORM_KEYS = ("model", "gate_model", "base_url")


def key_status(provider: str) -> str:
    load_dotenv()  # pick up a key added to .env while the app runs
    key_env = PROVIDERS[provider]["key_env"]
    if not key_env:
        return "No API key needed (local server)."
    if os.environ.get(key_env):
        return f"`{key_env}` is set."
    return f"**`{key_env}` is not set.** Add it to `.env` in the project folder; no restart needed."


# Model names with these words are not chat models (speech, images, embeddings, ...); left out of the dropdown.
NON_CHAT = ("embed", "tts", "image", "audio", "live", "transcribe", "translate", "lyria", "aqa", "banana",
            "imagen", "veo", "research", "computer-use", "robotics", "antigravity")


def chat_models(models: list[str]) -> list[str]:
    return [m for m in models if not any(word in m.lower() for word in NON_CHAT)]


def has_key(provider: str) -> bool:
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
# Layout and events
# ---------------------------------------------------------------------------
def build_panel(demo: gr.Blocks, cog: gr.Button) -> gr.Sidebar:
    """Create the settings sidebar (closed at first) inside `demo` and wire its events. The cog button or the
    panel's own edge arrow opens and closes it."""
    with gr.Sidebar(label="Settings", open=False, position="right", width=360) as panel:
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

    form = [model_in, gate_in, url_in, key_md]
    # Gradio's own progress overlay would cover the fields and show a countdown guessed from the last run;
    # test_connection writes its own "Testing..." line instead.
    quiet = dict(show_progress="hidden")
    # The cog toggles the panel; the panel's own arrow works too, so track its state from its events.
    panel_open = gr.State(False)
    panel.expand(lambda: True, outputs=panel_open, **quiet)
    panel.collapse(lambda: False, outputs=panel_open, **quiet)
    cog.click(lambda is_open: (gr.Sidebar(open=not is_open), not is_open), inputs=panel_open,
              outputs=[panel, panel_open], **quiet)
    demo.load(load_form, outputs=[provider_in, *form, settings_md], **quiet)
    provider_in.input(provider_form, inputs=provider_in, outputs=form, **quiet)
    test_btn.click(test_connection, inputs=[provider_in, model_in, url_in], outputs=[settings_md, model_in, gate_in], **quiet)
    save_btn.click(save_settings, inputs=[provider_in, model_in, gate_in, url_in], outputs=settings_md, **quiet)
    demo.load(load_thingiverse, outputs=[pick_in, pick_md], **quiet)
    pick_in.input(save_pick, inputs=pick_in, outputs=pick_md, **quiet)
    return panel
