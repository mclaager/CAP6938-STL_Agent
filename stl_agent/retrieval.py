"""RAG over the few-shot example library (data/examples.json).

Retrievers score every example against the request and share one `retrieve()` (top-k, a cap per
example class so the top-k stays diverse, and an optional similarity cutoff). Pick one with
`make_retriever(kind)` / `CAD_RETRIEVER`:

    tfidf       word TF-IDF over each example's request + keywords (default; no model needed)
    tfidf-char  TF-IDF over 3-5 letter chunks with numbers removed: "hexagonal" overlaps "hex",
                and sizes like "10mm" don't create false matches. No model needed.
    embed       meaning-based: a small local embedding model run with fastembed (ONNX, no PyTorch).
                The model (~130 MB) is downloaded once into the model cache, then works offline.
    hybrid      average of the tfidf-char and embed scores

eval/retrieval.py compares them against hand-labelled relevant examples (issue #20).

Scores are cosine similarities, but each method has its own range, so a `min_sim` cutoff tuned
for one does not carry over to another (see eval/retrieval.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import EMBED_MODEL, EXAMPLES_PATH, MODEL_CACHE_DIR, RAG_MIN_SIM, RETRIEVER, TOP_K


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

    def retrieve(self, query: str, k: int = TOP_K, min_sim: float = RAG_MIN_SIM) -> list[dict]:
        """Top-k examples by similarity, skipping any below `min_sim` (so it may return fewer, or none)."""
        sims = self.scores(query)
        picked, per_class = [], {}
        for i in sims.argsort()[::-1]:
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


class TfidfRetriever(BaseRetriever):
    """TF-IDF over each example's request text plus its `keywords` (not the code).

    Keywords are generic shape and operation words ("hexagon hex nut polygon") so a request can
    match an example that uses different words for the same kind of part. Word matching still
    misses synonyms that aren't in the keywords ("knob" vs a stopper); see EmbeddingRetriever.
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


class EmbeddingRetriever(BaseRetriever):
    """Cosine similarity between sentence embeddings from a local fastembed model.

    Numbers are removed first: in eval/retrieval.py that improved every model tried, because
    otherwise every request embeds as "a part with some millimetre sizes".
    """

    def __init__(self, examples: list[dict], max_per_class: int = 2, model: str = EMBED_MODEL,
                 cache_dir: Path = MODEL_CACHE_DIR):
        from fastembed import TextEmbedding

        super().__init__(examples, max_per_class)
        self.model = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
        # passage_embed / query_embed add the prefixes some models were trained with.
        docs = [strip_numbers(example_text(e)) for e in examples]
        self.matrix = self._normalize(np.array(list(self.model.passage_embed(docs))))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def scores(self, query: str) -> np.ndarray:
        q = self._normalize(np.array(list(self.model.query_embed([strip_numbers(query)])))[0])
        return self.matrix @ q


class HybridRetriever(BaseRetriever):
    """Average of letter-chunk TF-IDF and embedding scores: spelling and meaning both count."""

    def __init__(self, examples: list[dict], max_per_class: int = 2, weight: float = 0.5, **embed_kw):
        super().__init__(examples, max_per_class)
        self.tfidf = CharTfidfRetriever(examples, max_per_class)
        self.embed = EmbeddingRetriever(examples, max_per_class, **embed_kw)
        self.weight = weight  # share of the embedding score

    def scores(self, query: str) -> np.ndarray:
        return self.weight * self.embed.scores(query) + (1 - self.weight) * self.tfidf.scores(query)


RETRIEVERS = {"tfidf": TfidfRetriever, "tfidf-char": CharTfidfRetriever, "embed": EmbeddingRetriever,
              "hybrid": HybridRetriever}

# Kept for existing callers: the default retriever before CAD_RETRIEVER existed.
ExampleRetriever = TfidfRetriever


def make_retriever(kind: str = RETRIEVER, path: Path = EXAMPLES_PATH, **kw) -> BaseRetriever:
    if kind not in RETRIEVERS:
        raise ValueError(f"Unknown CAD_RETRIEVER {kind!r}; use one of {sorted(RETRIEVERS)}")
    return RETRIEVERS[kind].from_file(path, **kw)
