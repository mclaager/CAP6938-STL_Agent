"""RAG over the few-shot example library (data/examples.json)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import EXAMPLES_PATH, RAG_MIN_SIM, TOP_K


class ExampleRetriever:
    """TF-IDF similarity over example *request text only* (not code).

    TF-IDF needs no model download and works offline, which is plenty for a
    ~30-example library. To try dense embeddings later, replace `_vectorize`
    and `retrieve` with sentence-transformers; the interface stays the same.
    A per-class cap keeps the top-k diverse instead of three near-duplicates.
    """

    def __init__(self, examples: list[dict], max_per_class: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.examples = examples
        self.max_per_class = max_per_class
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([e["request"] for e in examples])

    @classmethod
    def from_file(cls, path: Path = EXAMPLES_PATH, **kw) -> "ExampleRetriever":
        return cls(json.loads(Path(path).read_text()), **kw)

    def retrieve(self, query: str, k: int = TOP_K, min_sim: float = RAG_MIN_SIM) -> list[dict]:
        """Top-k examples by similarity, skipping any below `min_sim` (so it may return fewer, or none)."""
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        picked, per_class = [], {}
        for i in order:
            if sims[i] < min_sim:
                break
            ex = self.examples[i]
            if per_class.get(ex["cls"], 0) >= self.max_per_class:
                continue
            picked.append(ex)
            per_class[ex["cls"]] = per_class.get(ex["cls"], 0) + 1
            if len(picked) == k:
                break
        return picked
