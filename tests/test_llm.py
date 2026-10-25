from stl_agent.llm import PROVIDERS, OpenAICompatLLM, make_llm


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    llm = make_llm(provider="ollama", model=None)
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model == PROVIDERS["ollama"]["default_model"]
    assert llm.pacer is None  # local server: no rate limit
    assert str(llm.client.base_url).startswith(PROVIDERS["ollama"]["base_url"].rstrip("/"))


def test_backends_have_the_llm_attributes(monkeypatch):
    from stl_agent.llm import LLM, AnthropicLLM, ScriptedLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # building the client sends no request
    for llm in (AnthropicLLM(), make_llm(provider="ollama", model=None), ScriptedLLM(["hi"])):
        assert isinstance(llm, LLM)
        for name in LLM.__annotations__:
            assert hasattr(llm, name), (type(llm).__name__, name)
