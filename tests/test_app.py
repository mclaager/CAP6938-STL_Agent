"""Chat flow through the app's handlers with the gate and build stubbed (no LLM calls)."""

import pytest

from stl_agent import app
from stl_agent.gate import decide_from_report

CRITICAL = {"scope": "in_scope", "spec": "stand",
            "unknowns": [{"param": "width", "default": "75", "critical": True, "question": "Phone width?"}]}
READY = {"scope": "in_scope", "spec": "stand 80mm", "name": "phone-stand",
         "unknowns": [{"param": "angle", "default": "60", "critical": False}]}


def last(gen):
    *_, final = gen
    return final


def test_clarify_then_build(monkeypatch, tmp_path):
    calls = []

    def fake_gate(conversation, rounds):
        calls.append(rounds)
        return decide_from_report(CRITICAL if rounds == 0 else READY, "x", rounds)

    def fake_pipeline(spec, log, name=None, **kw):
        log("working on " + spec)
        return str(tmp_path / f"{name}.stl"), "code", {"success": True}

    monkeypatch.setattr(app, "run_gate", fake_gate)
    monkeypatch.setattr(app, "run_pipeline", fake_pipeline)
    state = app.new_state()

    chat = last(app.on_message("phone stand", [], state, False))[1]
    assert "Phone width?" in chat[-1]["content"]
    assert state["rounds"] == 1 and state["spec"] is None

    chat = last(app.on_message("80mm", chat, state, False))[1]
    assert "stand 80mm" in chat[-1]["content"]
    assert state["spec"] == "stand 80mm" and state["name"] == "phone-stand"

    chat, state, _, log = last(app.on_build(chat, state))
    assert "phone-stand.stl" in chat[-1]["content"]
    assert "working on stand 80mm" in log
    assert state["result"]["stl"].endswith("phone-stand.stl")
    assert state["spec"] is None and state["rounds"] == 0
    assert calls == [0, 1]


# ---------------------------------------------------------------------------
# Thingiverse fallback for out-of-scope requests (#9)
# ---------------------------------------------------------------------------
HORSE = {"scope": "out_of_scope", "scope_reason": "organic shape", "spec": "horse", "name": "horse-statue",
         "search": "horse statue"}


def candidate(i):
    from stl_agent.sources.thingiverse import Candidate
    return Candidate(id=i, name=f"Horse {i}", url=f"https://www.thingiverse.com/thing:{i}",
                     thumbnail=f"https://cdn.example/{i}.jpg", author="maker", license="CC-BY")


def test_out_of_scope_shows_thingiverse_results(monkeypatch):
    from stl_agent.sources import thingiverse

    terms = []
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    monkeypatch.setattr(thingiverse, "search", lambda term: terms.append(term) or [candidate(1), candidate(2)])
    state = app.new_state()
    _, chat, state, build, gallery = last(app.on_message("a statue of a horse", [], state, False))
    assert terms == ["horse statue"]
    assert [c.id for c in state["candidates"]] == [1, 2]
    assert gallery.visible and "Thingiverse" in chat[-1]["content"]
    assert state["spec"] is None


def test_out_of_scope_without_token_explains(monkeypatch):
    from stl_agent.sources import thingiverse

    monkeypatch.delenv(thingiverse.TOKEN_VAR, raising=False)
    monkeypatch.setattr(app, "run_gate", lambda conv, rounds: decide_from_report(HORSE, "x", rounds))
    state = app.new_state()
    _, chat, state, _, gallery = last(app.on_message("a statue of a horse", [], state, False))
    assert "THINGIVERSE_TOKEN" in chat[-1]["content"] and "can't build" in chat[-1]["content"]
    assert not gallery.visible and state["candidates"] == []


def test_pick_downloads_and_credits(monkeypatch, tmp_path):
    def fake_fetch(c, log, name=None):
        log("Saved x")
        return str(tmp_path / "horse-statue.stl"), {"success": True, "source": {"other_stl_files": ["base.stl"]}}

    monkeypatch.setattr(app, "run_fetch", fake_fetch)
    state = app.new_state()
    state["candidates"], state["name"] = [candidate(1), candidate(2)], "horse-statue"

    class Evt:  # stands in for gr.SelectData
        index = 1

    chat, state, log = app.on_pick(Evt(), [], state)
    msg = chat[-1]["content"]
    assert "Horse 2" in msg and "maker" in msg and "CC-BY" in msg and "base.stl" in msg
    assert state["result"]["stl"].endswith("horse-statue.stl")
    assert "thing:2" in state["result"]["code"]
    assert "Saved x" in log


# ---------------------------------------------------------------------------
# Model settings panel
# ---------------------------------------------------------------------------
def test_save_settings_writes_file_and_drops_backends(monkeypatch):
    from stl_agent.config import settings

    reset = []
    monkeypatch.setattr(app, "reset_backends", lambda: reset.append(1))
    status = app.save_settings("ollama", "qwen3.5:9b", "", "http://gpu-box:11434/v1")
    assert status.startswith("Saved"), status
    s = settings()
    assert s.provider == "ollama"
    assert s.providers["ollama"]["model"] == "qwen3.5:9b"
    assert s.providers["ollama"]["base_url"] == "http://gpu-box:11434/v1"
    assert s.sources["providers.ollama.model"] == s.path.name
    assert reset == [1]


def test_save_settings_default_values_stay_commented(monkeypatch):
    from stl_agent.config import defaults, settings

    monkeypatch.setattr(app, "reset_backends", lambda: None)
    app.save_settings("ollama", "qwen3.5:9b", "", "")
    d = defaults()["providers"]["ollama"]
    # Back to the default value (and an empty address): the file uses the defaults again.
    app.save_settings("ollama", d["model"], "", "")
    s = settings()
    assert s.sources["providers.ollama.model"] == "default"
    assert s.sources["providers.ollama.base_url"] == "default"
    assert s.providers["ollama"]["base_url"] == d["base_url"]


def test_save_settings_rejects_bad_values(monkeypatch):
    monkeypatch.setattr(app, "reset_backends", lambda: None)
    assert app.save_settings("nope", "m", "", "").startswith("Not saved")


def test_provider_form_shows_current_values(monkeypatch):
    from stl_agent.config import settings

    def down(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(app, "list_models", down)  # offline: the current model is still offered

    model, gate, url, key = app.provider_form("ollama")
    assert gate.value == "" and gate.choices[0] == app.SAME_AS_MODEL  # default: the build model
    assert model.value == settings().providers["ollama"]["model"]
    assert url.value == settings().providers["ollama"]["base_url"]
    assert [c[0] for c in model.choices] == [settings().providers["ollama"]["model"]]
    assert "No API key needed" in key and "Couldn't list" in key


def test_key_status_names_missing_key(monkeypatch):
    monkeypatch.setattr(app, "load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not set" in app.key_status("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert "is set" in app.key_status("gemini")


def test_test_connection_reports_missing_model(monkeypatch):
    from stl_agent.llm import ScriptedLLM

    monkeypatch.setattr(app, "list_models", lambda provider, base_url: ["gemma4:12b", "qwen3.5:latest"])
    llm = ScriptedLLM(["OK", "OK"])
    llm.model = "llama9"
    monkeypatch.setattr(app, "make_llm", lambda *a, **kw: llm)
    status, dropdown, gate = last(app.test_connection("ollama", "llama9", ""))
    assert "not one of the server's 2 models" in status
    assert "ollama pull llama9" in status
    assert "'OK'" in status
    assert ("gemma4:12b", "gemma4:12b") in dropdown.choices  # the server's models become the choices
    assert gate.choices[0] == app.SAME_AS_MODEL and ("gemma4:12b", "gemma4:12b") in gate.choices

    status, *_ = last(app.test_connection("ollama", "qwen3.5", ""))  # Ollama lists it as qwen3.5:latest
    assert "not one of" not in status and "Connected" in status


def test_test_connection_survives_unreachable_server(monkeypatch):
    def down(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(app, "list_models", down)
    monkeypatch.setattr(app, "make_llm", down)
    status, *_ = last(app.test_connection("ollama", "gemma4:12b", ""))
    assert "Couldn't list" in status and "Request failed" in status


def test_test_connection_stops_without_key(monkeypatch):
    monkeypatch.setattr(app, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(app, "make_llm", lambda *a, **kw: pytest.fail("no request without a key"))
    steps = list(app.test_connection("anthropic", "claude-sonnet-5-5", ""))
    assert "Testing" in steps[0][0]
    assert "ANTHROPIC_API_KEY" in steps[-1][0] and "not set" in steps[-1][0]


def test_dropdown_lists_server_chat_models(monkeypatch):
    monkeypatch.setattr(app, "list_models", lambda provider, timeout=None: ["gemma4:12b", "nomic-embed-text", "qwen3.5:9b"])
    model, *_ = app.provider_form("ollama")
    names = [c[0] for c in model.choices]
    assert "qwen3.5:9b" in names and "nomic-embed-text" not in names


def test_test_connection_times_out_quickly(monkeypatch):
    from stl_agent.llm import ScriptedLLM

    used = {}

    class Slow(ScriptedLLM):
        def complete(self, system, messages):
            raise RuntimeError("gemini-3.8-flash: request timed out")

    def fake_make_llm(*a, **kw):
        used.update(kw)
        return Slow([])

    monkeypatch.setattr(app, "list_models", lambda provider, base_url: ["gemini-3.8-flash"])
    monkeypatch.setattr(app, "make_llm", fake_make_llm)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    status, *_ = last(app.test_connection("gemini", "gemini-3.8-flash", ""))
    assert used == {"timeout": app.TEST_TIMEOUT, "quick": True}
    assert f"No reply within {app.TEST_TIMEOUT} s" in status and "address and key work" in status
