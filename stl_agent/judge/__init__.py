"""Visual judge (issue #8): scores how well a generated part matches its request, from a render.

Used for evaluation (eval/judge.py), not in the app. Every backend returns a `Judgement` with a
0-1 score, so thresholds and calibration work the same for any backend.

Modules: base.py (types), llm_judge.py (multimodal LLM judge, versioned prompts).
A Jev backend (#12) would go in jev.py and return Jev's calibrated confidence as the score; whether
Jev accepts images, or would need a text description of the render, is still open.

Config via environment variables (see config.py):
    CAD_JUDGE            "llm" (default)
    CAD_JUDGE_PROVIDER   provider for the judge model (default: CAD_PROVIDER)
    CAD_JUDGE_MODEL      judge model (default: the provider's default_judge_model)
    CAD_JUDGE_THRESHOLD  score at which a part passes (set from `python -m eval.judge calibrate`)
"""

from __future__ import annotations

from ..config import JUDGE_KIND, JUDGE_MODEL, JUDGE_PROVIDER
from .base import Judge, Judgement
from .llm_judge import PROMPTS, LLMJudge, parse_judge_reply


def judge_model(provider: str = JUDGE_PROVIDER, model: str | None = JUDGE_MODEL) -> str:
    """The judge model that will be used: CAD_JUDGE_MODEL, else the provider's judge default."""
    from ..llm import PROVIDERS

    cfg = PROVIDERS.get(provider, {})
    return model or cfg.get("default_judge_model") or cfg.get("default_model") or "claude-sonnet-5-5"


def make_judge(kind: str = JUDGE_KIND, llm=None, model: str | None = JUDGE_MODEL, prompt: str = "v1",
               provider: str = JUDGE_PROVIDER) -> Judge:
    """`llm`: an LLM to use directly (tests); otherwise one is built for `provider` / `model`."""
    if kind != "llm":
        raise ValueError(f"Unknown CAD_JUDGE {kind!r}; only 'llm' exists so far.")
    if prompt not in PROMPTS:
        raise ValueError(f"Unknown judge prompt {prompt!r}; use one of {sorted(PROMPTS)}")
    name = judge_model(provider, model)
    if llm is None:
        from ..llm import make_llm

        llm = make_llm(provider=provider, model=name)
    return LLMJudge(llm, prompt=prompt, model_name=name)


__all__ = ["Judge", "Judgement", "LLMJudge", "PROMPTS", "judge_model", "make_judge", "parse_judge_reply"]
