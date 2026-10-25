from stl_agent.llm import PROVIDERS, OpenAICompatLLM, make_llm


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    llm = make_llm(provider="ollama", model=None)
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model == PROVIDERS["ollama"]["default_model"]
    assert llm.pacer is None  # local server: no rate limit
    assert str(llm.client.base_url).startswith(PROVIDERS["ollama"]["base_url"].rstrip("/"))

