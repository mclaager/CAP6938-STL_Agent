import tomllib

import pytest

from stl_agent import config
from stl_agent.config import (DEFAULTS_PATH, OPTIONAL, OPTIONAL_PROVIDER, PROVIDER_FIELDS, TOP_FIELDS,
                              commented_defaults, defaults, ignored_env_vars, override_settings,
                              reload_settings, set_setting, settings, settings_path, unset_setting)
from stl_agent.llm import PROVIDERS, make_llm, provider_config


@pytest.fixture
def write_settings():
    """Replace the settings file's text and reload."""
    def write(text: str):
        settings_path().write_text(text)
        return reload_settings()
    return write


# --- The defaults file is the template: it must list every setting the code knows -------------------
def test_defaults_file_matches_the_schema():
    data = defaults()
    lines = DEFAULTS_PATH.read_text().splitlines()
    listed = {(t, k) for _, t, k, _ in config._setting_lines(lines)}
    for key in TOP_FIELDS:
        assert ("", key) in listed, f"{key} is missing from settings.defaults.toml"
        assert key in data or key in OPTIONAL, f"{key} needs a default value"
    assert set(data["providers"]) == set(PROVIDERS)
    for name in PROVIDERS:
        for key in PROVIDER_FIELDS:
            assert (f"providers.{name}", key) in listed, f"providers.{name}.{key} is missing (a commented line is fine)"
        assert set(PROVIDER_FIELDS) - set(data["providers"][name]) <= OPTIONAL_PROVIDER


def test_new_settings_file_lists_everything_but_sets_nothing():
    s = settings()
    text = settings_path().read_text()
    assert text == commented_defaults()
    assert tomllib.loads(text) == {"providers": {name: {} for name in PROVIDERS}}  # only empty tables
    assert (s.provider, s.max_attempts, s.top_k, s.gate_max_rounds) == ("anthropic", 4, 3, 2)
    assert set(s.sources.values()) == {"default"}


def test_uncommented_lines_override_defaults(write_settings):
    s = write_settings('provider = "Ollama"\ntop_k = 1\n[providers.ollama]\nmodel = "qwen-local"\n')
    assert (s.provider, s.top_k, s.max_attempts) == ("ollama", 1, 4)
    assert s.sources["top_k"] == "settings.toml" and s.sources["max_attempts"] == "default"
    cfg = provider_config("ollama")
    assert (cfg["model"], cfg["gate_model"], cfg["judge_model"]) == ("qwen-local", "qwen-local", "qwen-local")
    assert cfg["base_url"] == defaults()["providers"]["ollama"]["base_url"]


def test_models_follow_the_selected_provider(write_settings):
    write_settings('provider = "gemini"\n')
    gem = defaults()["providers"]["gemini"]
    cfg = provider_config("gemini")
    assert (cfg["model"], cfg["judge_model"]) == (gem["model"], gem["judge_model"])
    assert cfg["gate_model"] == gem["model"]  # no gate model set: the gate uses the build model


def test_make_llm_uses_the_file(write_settings):
    write_settings('provider = "ollama"\n[providers.ollama]\nbase_url = "http://box:11434/v1"\nmodel = "m"\nrpm = 5\n')
    llm = make_llm()
    assert (llm.model, str(llm.client.base_url), llm.pacer.rpm) == ("m", "http://box:11434/v1/", 5)


@pytest.mark.parametrize("text, word", [
    ('provder = "ollama"', "provder"),
    ('top_k = "3"', "top_k"),
    ("gate_picks = 1", "gate_picks"),
    ("max_attempts = true", "max_attempts"),
    ('provider = "olama"', "olama"),
    ('[providers.ollama]\nmodle = "x"', "providers.ollama.modle"),
    ('[providers.olama]\nmodel = "x"', "providers.olama"),
    ('[providers.gemini]\napi_key = "x"', ".env"),
    ("top_k = ", "not valid TOML"),
])
def test_bad_settings_file_names_the_problem(write_settings, text, word):
    with pytest.raises(ValueError, match=word.replace(".", r"\.")):
        write_settings(text + "\n")


def test_missing_settings_are_added_commented_out(write_settings):
    write_settings('# my notes\ntop_k = 1\n\n[providers.ollama]\nmodel = "mine"\n')
    text = settings_path().read_text()
    assert text.startswith('# my notes\ntop_k = 1\n')  # the user's lines are untouched
    assert '# max_attempts = 4' in text and '# provider = "anthropic"' in text
    assert '[providers.gemini]' in text
    data = tomllib.loads(text)
    assert data["top_k"] == 1 and data["providers"]["ollama"] == {"model": "mine"}
    reload_settings()
    assert settings_path().read_text() == text  # nothing more to add the second time
    # each provider's lines land in that provider's table
    lines = text.splitlines()
    assert {t for _, t, k, _ in config._setting_lines(lines) if k == "base_url"} == {f"providers.{p}" for p in PROVIDERS}


def test_set_and_unset_edit_one_line_and_keep_comments():
    settings()  # creates the commented file
    before = settings_path().read_text()
    set_setting("provider", "ollama")
    set_setting("providers.ollama.model", "llama-local")
    set_setting("providers.ollama.rpm", 30)
    set_setting("gate_picks", True)
    text = settings_path().read_text()
    assert 'provider = "ollama"' in text.splitlines() and "gate_picks = true" in text.splitlines()
    assert "# max_attempts = 4" in text  # other lines kept
    assert len(text.splitlines()) == len(before.splitlines())  # uncommented in place, nothing added
    s = settings()
    assert (s.provider, s.gate_picks, provider_config("ollama")["model"], provider_config("ollama")["rpm"]) == \
        ("ollama", True, "llama-local", 30)
    assert s.sources["providers.ollama.model"] == "settings.toml"
    for key in ("provider", "providers.ollama.model", "providers.ollama.rpm", "gate_picks"):
        unset_setting(key)
    assert settings_path().read_text() == before
    assert settings().provider == "anthropic"


def test_set_setting_rejects_bad_values_and_forms_it_cannot_edit(write_settings):
    with pytest.raises(ValueError, match="top_k"):
        set_setting("top_k", "three")
    with pytest.raises(ValueError, match="unknown setting"):
        set_setting("providers.ollama", "x")
    write_settings('providers.ollama.model = "dotted"\n')
    with pytest.raises(ValueError, match="by hand"):
        set_setting("providers.ollama.model", "x")


def test_set_overrides_beat_the_file_and_are_not_saved(write_settings):
    write_settings("top_k = 2\n")
    text = settings_path().read_text()
    s = override_settings(["top_k=1", "providers.ollama.model=qwen3.5:9b", "gate_picks=true", "provider = ollama"])
    assert (s.top_k, s.gate_picks, s.provider, s.sources["top_k"]) == (1, True, "ollama", "--set")
    assert provider_config("ollama")["model"] == "qwen3.5:9b"
    assert settings_path().read_text() == text
    assert reload_settings().top_k == 1  # kept across reloads


@pytest.mark.parametrize("pair, word", [("top_k", "key=value"), ("topk=1", "topk"), ("top_k=x", "top_k"),
                                        ("providers.x.model=m", "providers.x")])
def test_bad_set_names_the_problem(pair, word):
    with pytest.raises(ValueError, match=word.replace(".", r"\.")):
        override_settings([pair])


def test_old_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("CAD_TOP_K", "5")
    assert reload_settings().top_k == 3
    assert "CAD_TOP_K" in ignored_env_vars() and "CAD_SETTINGS" not in ignored_env_vars()


def test_generate_cad_reads_max_attempts_from_settings(write_settings, tmp_path):
    from stl_agent.llm import ScriptedLLM
    from stl_agent.pipeline import generate_cad
    from stl_agent.retrieval import make_retriever

    write_settings("max_attempts = 2\n")
    llm = ScriptedLLM(["no code here"] * 5)
    res = generate_cad("a cube", llm, make_retriever(), out_dir=tmp_path / "run")
    assert not res.success and res.attempts == 2 and len(llm.replies) == 3


def test_bench_set_flag_reaches_settings(monkeypatch):
    from eval import bench

    seen = {}
    monkeypatch.setattr("stl_agent.llm.make_llm", lambda **kw: seen.setdefault("provider", settings().provider) and 1/0)
    with pytest.raises(ZeroDivisionError):
        bench.main(["run", "--set", "provider=ollama", "--cases", "none"])
    assert seen["provider"] == "ollama"
