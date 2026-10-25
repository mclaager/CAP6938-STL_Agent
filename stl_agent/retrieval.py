"""RAG over the few-shot example library (data/examples.json).

Retrievers score every example against the request and share one `retrieve()` (top-k, with a cap
per example class so the top-k stays diverse). Pick one with `make_retriever(kind)` / the `retriever` setting:

    tfidf       word TF-IDF over each example's request + keywords (default; no model needed)
    tfidf-char  TF-IDF over 3-5 letter chunks with numbers removed: "hexagonal" overlaps "hex",
                and sizes like "10mm" don't create false matches. No model needed.

eval/retrieval.py compares them against hand-labelled relevant examples (issue #20). A local
embedding model and a similarity cutoff were also tried and removed: neither beat these
(eval/results/retrieval-lib26.md, bench results for rag-t20 / rag-t30).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import EXAMPLES_PATH, settings


def example_text(e: dict) -> str:
    return f"{e['request']} {e.get('keywords', '')}".strip()


def strip_numbers(text: str) -> str:
    """Drop sizes ("12mm", "4.5", "90 degrees") and the "x" in "60 x 40": they say nothing about
    what kind of part it is, but shared numbers made unrelated parts look similar."""
    return re.sub(r"\b\d+(\.\d+)?\s*(mm|deg|degrees)?\b|\bx\b", " ", text.lower())


class BaseRetriever:
    def __init__(self, examples: list[dict], max_per_class: int = 2):
        self.examples = examples
        self.max_per_class = max_per_class

    @classmethod
    def from_file(cls, path: Path = EXAMPLES_PATH, **kw):
        return cls(json.loads(Path(path).read_text()), **kw)

    def scores(self, query: str) -> np.ndarray:
        """Similarity of `query` to every example, in library order."""
        raise NotImplementedError

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Top-k examples by similarity, at most `max_per_class` from any one example class."""
        k = settings().top_k if k is None else k
        sims = self.scores(query)
        picked, per_class = [], {}
        for i in sims.argsort()[::-1]:
            ex = self.examples[i]
            if per_class.get(ex["cls"], 0) >= self.max_per_class:
                continue
            picked.append(ex)
            per_class[ex["cls"]] = per_class.get(ex["cls"], 0) + 1
            if len(picked) == k:
                break
        return picked


class TfidfRetriever(BaseRetriever):
    """TF-IDF over each example's request text plus its `keywords` (not the code).

    Keywords are generic shape and operation words ("hexagon hex nut polygon") so a request can
    match an example that uses different words for the same kind of part. Word matching still
    misses synonyms that aren't in the keywords ("knob" vs a stopper).
    """

    def __init__(self, examples: list[dict], max_per_class: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        super().__init__(examples, max_per_class)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([example_text(e) for e in examples])

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([query]).T).toarray().ravel()


class CharTfidfRetriever(TfidfRetriever):
    """TF-IDF over 3-5 letter chunks within words, with numbers removed. Best of the no-model
    methods in eval/retrieval.py, and as fast as word TF-IDF (under 1 ms per request)."""

    def __init__(self, examples: list[dict], max_per_class: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        BaseRetriever.__init__(self, examples, max_per_class)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, preprocessor=strip_numbers)
        self.matrix = self.vec.fit_transform([example_text(e) for e in examples])


RETRIEVERS = {"tfidf": TfidfRetriever, "tfidf-char": CharTfidfRetriever}

# Kept for existing callers: the default retriever before the retriever setting existed.
ExampleRetriever = TfidfRetriever


def make_retriever(kind: str | None = None, path: Path = EXAMPLES_PATH, **kw) -> BaseRetriever:
    kind = (kind or settings().retriever).lower()
    if kind not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {kind!r}; use one of {sorted(RETRIEVERS)}")
    return RETRIEVERS[kind].from_file(path, **kw)
