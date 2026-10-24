"""Judge types: what every judge backend returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Judgement:
    score: float | None = None  # 0-1, how well the part matches the request; None when the judge failed
    raw_score: float | None = None  # the backend's own scale (the LLM judge uses 1-10)
    matches: list[str] = field(default_factory=list)  # requested features the judge saw
    problems: list[str] = field(default_factory=list)  # missing, wrong or extra features
    reply: str = ""  # raw backend reply, kept for error analysis
    error: str | None = None  # set when the reply couldn't be used; such rows are left out of pass rates

    def passed(self, threshold: float) -> bool | None:
        return None if self.score is None else self.score >= threshold

    def to_dict(self) -> dict:
        return asdict(self)


class Judge(Protocol):
    name: str  # recorded with every result, e.g. "llm:gemini-3.1-flash-lite"

    def judge(self, request: str, image_png: bytes, facts: dict) -> Judgement:
        """`facts`: things the image can't show, e.g. {"bbox_mm": [60, 40, 5]}."""
        ...
