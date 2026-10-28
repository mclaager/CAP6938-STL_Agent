"""The settings panel (⚙): model settings saved to the settings file, Test connection, Thingiverse pick mode.
Server calls are stubbed (no LLM requests)."""

import pytest

from stl_agent import settings_panel as panel


def last(gen):
    *_, final = gen
    return final


# ---------------------------------------------------------------------------
# Model settings panel
# ---------------------------------------------------------------------------
def test_save_settings_writes_file_and_drops_backends(monkeypatch):
    from stl_agent.config import settings

    reset = []
    monkeypatch.setattr(panel, "reset_backends", lambda: reset.append(1))
    status = panel.save_settings("ollama", "qwen3.5:9b", "", "http://gpu-box:11434/v1")
    assert status.startswith("Saved"), status
    s = settings()
    assert s.provider == "ollama"
    assert s.providers["ollama"]["model"] == "qwen3.5:9b"
    assert s.providers["ollama"]["base_url"] == "http://gpu-box:11434/v1"
    assert s.sources["providers.ollama.model"] == s.path.name
    assert reset == [1]


def test_save_settings_default_values_stay_commented(monkeypatch):
    from stl_agent.config import defaults, settings

    monkeypatch.setattr(panel, "reset_backends", lambda: None)
    panel.save_settings("ollama", "qwen3.5:9b", "", "")
    d = defaults()["providers"]["ollama"]
    # Back to the default value (and an empty address): the file uses the defaults again.
    panel.save_settings("ollama", d["model"], "", "")
    s = settings()
    assert s.sources["providers.ollama.model"] == "default"
    assert s.sources["providers.ollama.base_url"] == "default"
    assert s.providers["ollama"]["base_url"] == d["base_url"]


def test_save_settings_rejects_bad_values(monkeypatch):
    monkeypatch.setattr(panel, "reset_backends", lambda: None)
    assert panel.save_settings("nope", "m", "", "").startswith("Not saved")


def test_provider_form_shows_current_values(monkeypatch):
    from stl_agent.config import settings

    def down(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(panel, "list_models", down)  # offline: the current model is still offered

    model, gate, url, key = panel.provider_form("ollama")
    assert gate.value == "" and gate.choices[0] == panel.SAME_AS_MODEL  # default: the build model
    assert model.value == settings().providers["ollama"]["model"]
    assert url.value == settings().providers["ollama"]["base_url"]
    assert [c[0] for c in model.choices] == [settings().providers["ollama"]["model"]]
    assert "No API key needed" in key and "Couldn't list" in key


def test_key_status_names_missing_key(monkeypatch):
    monkeypatch.setattr(panel, "load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not set" in panel.key_status("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert "is set" in panel.key_status("gemini")


def test_test_connection_reports_missing_model(monkeypatch):
    from stl_agent.llm import ScriptedLLM

    monkeypatch.setattr(panel, "list_models", lambda provider, base_url: ["gemma4:12b", "qwen3.5:latest"])
    llm = ScriptedLLM(["OK", "OK"])
    llm.model = "llama9"
    monkeypatch.setattr(panel, "make_llm", lambda *a, **kw: llm)
    status, dropdown, gate = last(panel.test_connection("ollama", "llama9", ""))
    assert "not one of the server's 2 models" in status
    assert "ollama pull llama9" in status
    assert "'OK'" in status
    assert ("gemma4:12b", "gemma4:12b") in dropdown.choices  # the server's models become the choices
    assert gate.choices[0] == panel.SAME_AS_MODEL and ("gemma4:12b", "gemma4:12b") in gate.choices

    status, *_ = last(panel.test_connection("ollama", "qwen3.5", ""))  # Ollama lists it as qwen3.5:latest
    assert "not one of" not in status and "Connected" in status


def test_test_connection_survives_unreachable_server(monkeypatch):
    def down(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(panel, "list_models", down)
    monkeypatch.setattr(panel, "make_llm", down)
    status, *_ = last(panel.test_connection("ollama", "gemma4:12b", ""))
    assert "Couldn't list" in status and "Request failed" in status


def test_test_connection_stops_without_key(monkeypatch):
    monkeypatch.setattr(panel, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(panel, "make_llm", lambda *a, **kw: pytest.fail("no request without a key"))
    steps = list(panel.test_connection("anthropic", "claude-sonnet-5-5", ""))
    assert "Testing" in steps[0][0]
    assert "ANTHROPIC_API_KEY" in steps[-1][0] and "not set" in steps[-1][0]


def test_dropdown_lists_server_chat_models(monkeypatch):
    monkeypatch.setattr(panel, "list_models", lambda provider, timeout=None: ["gemma4:12b", "nomic-embed-text", "qwen3.5:9b"])
    model, *_ = panel.provider_form("ollama")
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

    monkeypatch.setattr(panel, "list_models", lambda provider, base_url: ["gemini-3.8-flash"])
    monkeypatch.setattr(panel, "make_llm", fake_make_llm)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    status, *_ = last(panel.test_connection("gemini", "gemini-3.8-flash", ""))
    assert used == {"timeout": panel.TEST_TIMEOUT, "quick": True}
    assert f"No reply within {panel.TEST_TIMEOUT} s" in status and "address and key work" in status


def test_pick_mode_saves_and_default_stays_commented():
    from stl_agent.config import settings

    assert panel.load_thingiverse()[0] == "manual"
    assert panel.save_pick("auto").startswith("Saved")
    s = settings()
    assert s.thingiverse_pick == "auto" and s.sources["thingiverse_pick"] == s.path.name
    panel.save_pick("manual")  # back to the default: the line is commented out again
    assert settings().sources["thingiverse_pick"] == "default"
    assert panel.save_pick("sometimes").startswith("Not saved")


def test_thingiverse_status_mentions_token(monkeypatch):
    from stl_agent.sources import thingiverse

    monkeypatch.setattr(panel, "load_dotenv", lambda: None)
    monkeypatch.delenv(thingiverse.TOKEN_VAR, raising=False)
    assert "is not set" in panel.thingiverse_status()
    monkeypatch.setenv(thingiverse.TOKEN_VAR, "t")
    assert "is set" in panel.thingiverse_status()
