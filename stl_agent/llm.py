"""LLM backends. Every backend follows the `LLM` interface below; `make_llm` builds one from settings.

A message's "content" is either a string or a list of parts, so images can be sent to vision models:
    {"type": "text", "text": "..."}   and   {"type": "image", "png": b"..."}
Each backend converts the parts to its own API format.
"""

from __future__ import annotations

import base64
import os
import re
import time
from typing import Protocol, runtime_checkable

from .config import Settings, settings


@runtime_checkable
class LLM(Protocol):
    """What the pipeline, gate and judge expect from a backend."""

    input_tokens: int  # running totals over all requests, for cost reports
    output_tokens: int
    last_truncated: bool  # the last reply was cut off by the output token limit
    last_reasoning: str = ""  # the model's reasoning before the last reply, when the provider returns it

    def complete(self, system: str, messages: list[dict]) -> str:
        """Send one request and return the reply text."""
        ...


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None, max_tokens: int = 2048, temperature: float | None = None,
                 base_url: str | None = None, timeout: float = 300.0, max_retries: int = 2):
        import anthropic

        # Reads ANTHROPIC_API_KEY, and ANTHROPIC_BASE_URL when no base_url is given.
        kw = {"base_url": base_url} if base_url else {}
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries, **kw)
        self.model = model or provider_config("anthropic")["model"]
        self.max_tokens, self.temperature = max_tokens, temperature
        self.input_tokens = self.output_tokens = 0
        self.last_truncated = False

    def complete(self, system: str, messages: list[dict]) -> str:
        messages = [{**m, "content": anthropic_content(m["content"])} for m in messages]
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        # Newer SDK/model versions don't accept sampling params; only send if explicitly requested.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        self.last_truncated = resp.stop_reason == "max_tokens"
        self.last_reasoning = "".join(getattr(b, "thinking", "") for b in resp.content if getattr(b, "type", "") == "thinking")
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatLLM(LLM):
    """Any provider with an OpenAI-style /chat/completions endpoint (Gemini, Groq,
    OpenRouter, Ollama, ...). Uses plain httpx so no extra SDK is needed."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        # Gemini 3.x "thinks" before answering and those hidden tokens count against this limit;
        # at 4096 replies were regularly cut off mid-code. Only tokens actually used are billed.
        max_tokens: int = 16384,
        temperature: float | None = None,
        # Long thinking replies can take minutes; at 120 s some benchmark requests timed out.
        timeout: float = 300.0,
        max_retries: int = 4,
        rpm: int | None = None,
    ):
        import httpx

        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.max_retries = max_retries
        self.pacer = RateLimiter(rpm) if rpm else None
        self.input_tokens = self.output_tokens = 0
        self.last_truncated = False

    def complete(self, system: str, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         *({**m, "content": openai_content(m["content"])} for m in messages)],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature

        import httpx

        # Free tiers rate-limit hard (429) and are sometimes overloaded (503); back off and retry.
        # A timed-out request is retried once only: the provider may still count it against the quota.
        timed_out, resp = False, None
        for attempt in range(self.max_retries + 1):
            if self.pacer:
                self.pacer.wait()
            try:
                resp = self.client.post("chat/completions", json=body)
            except httpx.TimeoutException:
                if timed_out:
                    raise RuntimeError(f"{self.model}: request timed out twice") from None
                timed_out = True
                continue
            if resp.status_code not in (429, 500, 502, 503) or attempt == self.max_retries:
                break
            if resp.status_code == 429 and "quota" in resp.text.lower():
                # Per-minute limits ask for a short wait: wait and retry. Daily limits ask for hours:
                # retrying just burns time, so stop.
                wait = _retry_delay(resp.text)
                if wait is None or wait > MAX_QUOTA_WAIT:
                    raise RuntimeError(f"{self.model}: free-tier quota exhausted. {_quota_hint(resp.text)}")
                time.sleep(wait + 1)
                continue
            retry_after = resp.headers.get("retry-after", "")
            time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1))
        if resp is None:
            raise RuntimeError(f"{self.model}: request timed out")
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.model} request failed ({resp.status_code}): {resp.text[:500]}")

        data = resp.json()
        usage = data.get("usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        choice = data["choices"][0]
        self.last_truncated = choice.get("finish_reason") == "length"
        msg = choice["message"]
        self.last_reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""  # Ollama / others
        return msg.get("content") or ""


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def anthropic_content(content):
    """Message content in Anthropic's format (strings pass through)."""
    if isinstance(content, str):
        return content
    return [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(p["png"])}}
            if p["type"] == "image" else p for p in content]


def openai_content(content):
    """Message content in the OpenAI chat format (strings pass through)."""
    if isinstance(content, str):
        return content
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(p["png"])}}
            if p["type"] == "image" else p for p in content]


class RateLimiter:
    """Allows at most `rpm` calls in any 60-second window by sleeping before a call when needed."""

    def __init__(self, rpm: int, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.clock, self.sleep = rpm, clock, sleep
        self.sent: list[float] = []

    def wait(self) -> None:
        now = self.clock()
        self.sent = [t for t in self.sent if now - t < 60]
        if len(self.sent) >= self.rpm:
            self.sleep(60 - (now - self.sent[0]) + 0.5)
            now = self.clock()
            self.sent = [t for t in self.sent if now - t < 60]
        self.sent.append(now)


MAX_QUOTA_WAIT = 90  # seconds; longer suggested waits mean a daily quota is used up


def _retry_delay(text: str) -> float | None:
    """Seconds the provider asks us to wait, from "retry in 47.2s" or "retryDelay": "47s"."""
    m = re.search(r"retry in ([\d.]+)s\b", text) or re.search(r'"retryDelay":\s*"([\d.]+)s"', text)
    return float(m.group(1)) if m else None


def _quota_hint(text: str) -> str:
    m = re.search(r"Quota exceeded for metric: [^,]*, limit: (\d+)[^\n]*\n?[^\n]*?retry in ([\dhms.]+)", text)
    if m:
        return (f"Limit is {m.group(1)} requests; retry in {m.group(2).split('.')[0]}s, or switch model "
                "(providers.gemini.model in settings.toml, or --model).")
    return "Try again later, or switch model (providers.gemini.model in settings.toml, or --model)."


# What the code needs to know about each provider. Everything a user may change (address, models, rpm)
# is in data/settings.defaults.toml and settings.toml instead. Add more OpenAI-compatible providers here
# and in the defaults file.
PROVIDERS = {
    "anthropic": dict(key_env="ANTHROPIC_API_KEY"),
    "gemini": dict(key_env="GEMINI_API_KEY"),
    "ollama": dict(key_env=None),  # local server, no key
}


def provider_config(name: str, s: Settings | None = None) -> dict:
    """One provider's settings: key_env, base_url, model, gate_model, judge_model, rpm, timeout.
    Unset gate and judge models fall back to the build model; unset base_url / rpm / timeout are None."""
    s = s or settings()
    name = name.lower()
    if name not in PROVIDERS or name not in s.providers:
        raise ValueError(f"Unknown provider {name!r}; use one of {sorted(PROVIDERS)}")
    p = s.providers[name]
    model = p["model"]
    return dict(key_env=PROVIDERS[name]["key_env"], base_url=p.get("base_url"), model=model,
                gate_model=p.get("gate_model") or model, judge_model=p.get("judge_model") or model,
                rpm=p.get("rpm"), timeout=p.get("timeout"))


def make_llm(provider: str | None = None, model: str | None = None, base_url: str | None = None,
             timeout: float | None = None, quick: bool = False) -> LLM:
    """Build an LLM backend. Arguments left as None come from the current settings (see provider_config).
    `quick`: no retries, for a connection test that should fail fast (use with a short `timeout`)."""
    provider = (provider or settings().provider).lower()
    cfg = provider_config(provider)
    model, base_url = model or cfg["model"], base_url or cfg["base_url"]
    if provider == "anthropic":
        api_key(provider)  # a clear error now instead of the SDK's at the first request
    timeout = timeout or cfg["timeout"]
    kw = {"timeout": float(timeout)} if timeout else {}
    if quick:
        kw["max_retries"] = 0
    if provider == "anthropic":
        return AnthropicLLM(model=model, base_url=base_url, **kw)
    return OpenAICompatLLM(base_url, api_key(provider), model, rpm=cfg["rpm"] or None, **kw)


def api_key(provider: str) -> str:
    """The provider's API key from the environment (.env); "none" for local servers, which ignore it."""
    key_env = PROVIDERS[provider.lower()]["key_env"]
    if not key_env:
        return "none"
    if not os.environ.get(key_env):
        raise RuntimeError(f"Provider {provider} needs {key_env} in .env (or the environment).")
    return os.environ[key_env]


def list_models(provider: str | None = None, base_url: str | None = None, timeout: float = 10.0) -> list[str]:
    """Model names the provider's server offers (for Ollama, the models pulled on that machine).
    Arguments left as None come from the current settings. Raises if the server can't be reached."""
    provider = (provider or settings().provider).lower()
    base_url = base_url or provider_config(provider)["base_url"]
    if provider == "anthropic":
        import anthropic

        kw = {"base_url": base_url} if base_url else {}
        client = anthropic.Anthropic(api_key=api_key(provider), timeout=timeout, **kw)
        return sorted(m.id for m in client.models.list(limit=100))
    import httpx

    resp = httpx.get(base_url.rstrip("/") + "/models", timeout=timeout,
                     headers={"Authorization": f"Bearer {api_key(provider)}"})
    if resp.status_code >= 400:
        raise RuntimeError(f"listing models failed ({resp.status_code}): {resp.text[:300]}")
    # Gemini names models "models/gemini-..."; requests use the name without the prefix.
    return sorted(m["id"].removeprefix("models/") for m in resp.json().get("data", []))


class ScriptedLLM(LLM):
    """Returns canned replies in order. For tests and offline demos."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), []
        self.input_tokens = self.output_tokens = 0
        self.last_truncated = False

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)
