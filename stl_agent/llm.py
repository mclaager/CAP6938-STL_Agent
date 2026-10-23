"""LLM backends. Anything with `.complete(system, messages) -> str` works as an LLM."""

from __future__ import annotations

import os
import re
import time

from .config import MODEL, PROVIDER


class AnthropicLLM:
    def __init__(self, model: str | None = MODEL, max_tokens: int = 2048, temperature: float | None = None):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or "claude-sonnet-5-5"
        self.max_tokens, self.temperature = max_tokens, temperature
        self.input_tokens = self.output_tokens = 0

    def complete(self, system: str, messages: list[dict]) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        # Newer SDK/model versions don't accept sampling params; only send if explicitly requested.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatLLM:
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
            "messages": [{"role": "system", "content": system}, *messages],
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
        return choice["message"].get("content") or ""


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
        return f"Limit is {m.group(1)} requests; retry in {m.group(2).split('.')[0]}s, or set CAD_MODEL to another model."
    return "Try again later, or set CAD_MODEL to another model."


# name -> how to build it. Add more OpenAI-compatible providers here.
PROVIDERS = {
    "gemini": dict(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-3.8-flash",
        default_gate_model="gemini-3.6-flash",  # separate daily quota from the build model
        # Free tier allows 15 requests/minute per model. Google counts requests at a different moment
        # than when we send them, so 14 still showed up as 17 in one minute; keep a wide margin.
        default_rpm=10,
    ),
}


def make_llm(provider: str = PROVIDER, model: str | None = MODEL):
    """Build the LLM backend selected by CAD_PROVIDER / CAD_MODEL."""
    if provider == "anthropic":
        return AnthropicLLM(model=model)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown CAD_PROVIDER {provider!r}; use 'anthropic' or one of {sorted(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"CAD_PROVIDER={provider} needs {cfg['key_env']} set (env or .env file).")
    rpm = int(os.environ.get("CAD_RPM") or cfg.get("default_rpm") or 0) or None
    return OpenAICompatLLM(cfg["base_url"], api_key, model or cfg["default_model"], rpm=rpm)


class ScriptedLLM:
    """Returns canned replies in order. For tests and offline demos."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), []

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)
