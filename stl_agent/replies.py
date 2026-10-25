"""Helpers for reading structured data out of model replies (shared by the gate and the judge)."""

from __future__ import annotations

import json
import re

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S | re.I)


def parse_json_object(text: str) -> dict:
    """Pull a JSON object out of a model reply (fenced block preferred, else the outermost braces).
    Raises ValueError (or json.JSONDecodeError) when there is none."""
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data
