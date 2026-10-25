"""Visual judge (issue #8): scores how well a generated part matches its request, from a render.

Used for evaluation (eval/judge.py), not in the app. Every backend returns a `Judgement` with a
0-1 score, so thresholds and calibration work the same for any backend.

Modules: base.py (types), llm_judge.py (multimodal LLM judge). Its system prompts are text files in
prompts/ (one per version, see prompts/README.md), not in the code.
A Jev backend (#12) would go in jev.py and return Jev's calibrated confidence as the score; whether
Jev accepts images, or would need a text description of the render, is still open.

Settings (settings.toml, see config.py):
    judge                       "llm" (default)
    judge_provider              provider for the judge model (default: the build provider)
    providers.<p>.judge_model   judge model (default: that provider's build model)
    judge_threshold             score at which a part passes (set from `python -m eval.judge calibrate`)
"""

from __future__ import annotations

from ..config import settings
from .base import Judge, Judgement
from .llm_judge import PROMPTS, LLMJudge, parse_judge_reply


def judge_provider(provider: str | None = None) -> str:
    s = settings()
    return (provider or s.judge_provider or s.provider).lower()


def judge_model(provider: str | None = None, model: str | None = None) -> str:
    """The judge model that will be used: `model`, else the judge model from settings (see llm.provider_config)."""
    from ..llm import provider_config

    return model or provider_config(judge_provider(provider))["judge_model"]


DEFAULT_PROMPT = "v2"  # matches render v2; v1 was written for render v1's angled underside view


def make_judge(kind: str | None = None, llm=None, model: str | None = None, prompt: str = DEFAULT_PROMPT,
               provider: str | None = None) -> Judge:
    """`llm`: an LLM to use directly (tests); otherwise one is built for `provider` / `model`.
    Arguments left as None come from the current settings."""
    kind = (kind or settings().judge).lower()
    if kind != "llm":
        raise ValueError(f"Unknown judge {kind!r}; only 'llm' exists so far.")
    if prompt not in PROMPTS:
        raise ValueError(f"Unknown judge prompt {prompt!r}; use one of {sorted(PROMPTS)}")
    name = judge_model(provider, model)
    if llm is None:
        from ..llm import make_llm

        llm = make_llm(provider=judge_provider(provider), model=name)
    return LLMJudge(llm, prompt=prompt, model_name=name)


__all__ = ["DEFAULT_PROMPT", "Judge", "Judgement", "LLMJudge", "PROMPTS", "judge_model", "make_judge",
           "parse_judge_reply"]
