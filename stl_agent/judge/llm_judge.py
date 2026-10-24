"""Judge backed by a multimodal LLM: one request with the part's render and the original request."""

from __future__ import annotations

import json

from ..llm import parse_json_object
from .base import Judgement

# Versioned so studies can compare prompts; judged rows record the version. Add a key, don't edit one.
# v1 was written for render v1 (front/right/top + iso front/iso back/from below).
PROMPTS = {
    "v1": """You check whether a 3D-printable part matches the request it was made from.

You see one image with six views of the same part, labelled in each corner: front, right and top
(flat views), then iso front, iso back and from below (angled 3D views; the last two show the back and
the underside). Each view is zoomed to fit, so sizes can't be compared between views, and there is no
scale; the measured overall size is given in the message.

Judge the SHAPE: are the requested features there, in the right number, kind and place (holes, slots,
pockets, walls, fillets, text, angles, which side is open)? Is anything major missing or added?
Use the measured size for overall dimensions; don't guess small dimensions from the picture.
Ignore color, shading, render quality and how the part is oriented in the views.

Score 1-10:
  9-10  matches the request
  7-8   matches with small deviations (a slightly different proportion or spacing)
  4-6   recognisably the requested part, but a requested feature is wrong or missing
  1-3   a different object, or most requested features are missing

Reply with only a JSON object, problems before the score:
{"matches": ["requested features you can see"], "problems": ["missing, wrong or extra features"], "score": 1-10}""",
}

# v2: written for render v2 (flat bottom view instead of v1's angled "from below" view). Explains flat vs
# angled views, after v1 read through-holes as blind pockets in the angled underside view.
PROMPTS["v2"] = """You check whether a 3D-printable part matches the request it was made from.

You see one image with six views of the same part, labelled in each corner:
- front, right, top and bottom are FLAT views, looking straight at the part. A hole that goes all the way
  through shows the white background through it; a blind hole or pocket shows a shaded floor instead.
- "iso front (angled)" and "iso back (angled)" are 3D views from above at an angle, for the overall shape
  and the back side. In angled views you look into holes at a slant, so even a through-hole looks closed:
  decide whether holes go through from the flat views, never from the angled ones.
Each view is zoomed to fit, so sizes can't be compared between views, and there is no scale; the measured
overall size is given in the message.

Judge the SHAPE: are the requested features there, in the right number, kind and place (holes, slots,
pockets, walls, fillets, text, angles, which side is open)? Is anything major missing or added?
Use the measured size for overall dimensions; don't guess small dimensions from the picture.
Ignore color, shading, render quality and how the part is oriented in the views.

Score 1-10:
  9-10  matches the request
  7-8   matches with small deviations (a slightly different proportion or spacing)
  4-6   recognisably the requested part, but a requested feature is wrong or missing
  1-3   a different object, or most requested features are missing

Reply with only a JSON object, problems before the score:
{"matches": ["requested features you can see"], "problems": ["missing, wrong or extra features"], "score": 1-10}"""

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
