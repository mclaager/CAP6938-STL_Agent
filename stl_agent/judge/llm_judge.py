"""Judge backed by a multimodal LLM: one request with the part's render and the original request."""

from __future__ import annotations

import json
from pathlib import Path

from ..llm import parse_json_object
from .base import Judgement

# System prompts live in prompts/<version>.txt (see prompts/README.md). Judged rows record the version,
# so released files are never edited; a change is a new file.
PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPTS = {f.stem: f.read_text() for f in sorted(PROMPT_DIR.glob("*.txt"))}
# The render version each prompt describes; eval/judge.py warns when they don't match.
RENDER_FOR_PROMPT = {"v1": "v1", "v2": "v2"}

SCALE = (1, 10)


def user_message(request: str, image_png: bytes, facts: dict) -> list[dict]:
    text = f"Request: {request}"
    if facts.get("bbox_mm"):
        text += "\nMeasured overall size: " + " x ".join(f"{v:g}" for v in facts["bbox_mm"]) + " mm"
    return [{"type": "text", "text": text}, {"type": "image", "png": image_png}]


def parse_judge_reply(reply: str) -> Judgement:
    try:
        data = parse_json_object(reply)
        raw = float(data["score"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return Judgement(reply=reply, error="reply had no JSON object with a numeric score")
    lo, hi = SCALE
    if not lo <= raw <= hi:
        return Judgement(raw_score=raw, reply=reply, error=f"score {raw:g} outside {lo}-{hi}")
    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
    return Judgement(score=round((raw - lo) / (hi - lo), 4), raw_score=raw, reply=reply,
                     matches=as_list(data.get("matches")), problems=as_list(data.get("problems")))


class LLMJudge:
    def __init__(self, llm, prompt: str = "v2", model_name: str = ""):
        self.llm, self.prompt = llm, prompt
        self.name = f"llm:{model_name or getattr(llm, 'model', 'unknown')}"

    def judge(self, request: str, image_png: bytes, facts: dict) -> Judgement:
        reply = self.llm.complete(PROMPTS[self.prompt], [{"role": "user", "content": user_message(request, image_png, facts)}])
        return parse_judge_reply(reply)
