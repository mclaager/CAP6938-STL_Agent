from stl_agent.llm import OpenAICompatLLM, make_llm, provider_config


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    llm = make_llm(provider="ollama", model=None)
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model == provider_config("ollama")["model"]
    assert llm.pacer is None  # local server: no rate limit
    assert str(llm.client.base_url).startswith(provider_config("ollama")["base_url"].rstrip("/"))


def test_backends_have_the_llm_attributes(monkeypatch):
    from stl_agent.llm import LLM, AnthropicLLM, ScriptedLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # building the client sends no request
    for llm in (AnthropicLLM(), make_llm(provider="ollama", model=None), ScriptedLLM(["hi"])):
        assert isinstance(llm, LLM)
        for name in LLM.__annotations__:
            assert hasattr(llm, name), (type(llm).__name__, name)


def test_list_models_strips_gemini_prefix(monkeypatch):
    import httpx

    from stl_agent.llm import list_models

    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "models/gemini-3.6-flash"}, {"id": "models/gemini-3.1-flash-lite"}]}

    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return Resp()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "get", fake_get)
    assert list_models("gemini") == ["gemini-3.1-flash-lite", "gemini-3.6-flash"]
    assert seen["url"].endswith("/models")


def test_quick_llm_does_not_retry(monkeypatch):
    from stl_agent.llm import AnthropicLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = make_llm(provider="ollama", timeout=5, quick=True)
    assert llm.max_retries == 0 and llm.client.timeout.read == 5
    llm = make_llm(provider="anthropic", timeout=5, quick=True)
    assert isinstance(llm, AnthropicLLM) and llm.client.max_retries == 0 and llm.client.timeout == 5
    assert make_llm(provider="anthropic").client.timeout == 300  # the timeout setting's default


def test_anthropic_without_key_says_so(monkeypatch):
    import pytest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm(provider="anthropic")
