"""Paths and settings. Reads a `.env` file in the repo root; real environment variables win.

Environment variables:
    CAD_PROVIDER        "anthropic" (default) or "gemini"
    ANTHROPIC_API_KEY   required when CAD_PROVIDER=anthropic
    GEMINI_API_KEY      required when CAD_PROVIDER=gemini (free key: aistudio.google.com)
    CAD_MODEL           model id (default depends on provider, see llm.PROVIDERS)
    CAD_GATE_MODEL      model for the intake gate (default: the provider's gate model, else CAD_MODEL).
                        A separate model has its own free-tier quota, so gate checks don't use up builds.
    CAD_MAX_ATTEMPTS    generation + repair attempts (default 4)
    CAD_TOP_K           RAG examples per prompt (default 3)
    CAD_GATE_PICKS      1 = the intake gate picks the RAG examples from a library index; when it picks
                        none, the retriever runs on the spec (default 0 until tested on all requests, #20)
    CAD_RETRIEVER       "tfidf" (default), "tfidf-char", "embed" (local fastembed model) or "hybrid"; see retrieval.py
    CAD_EMBED_MODEL     fastembed model for embed/hybrid (default snowflake/snowflake-arctic-embed-s, ~130 MB)
    CAD_MODEL_CACHE     where downloaded models are kept (default ~/.cache/stl_agent)
    CAD_RAG_MIN_SIM     skip examples with TF-IDF similarity below this (default 0 = always use top k).
                        A cutoff helped with the old 9-example library but not with the 26-example one.
    CAD_RPM             max LLM requests per minute per model (Gemini default 10; free tier allows 15)
    CAD_GATE, CAD_GATE_MAX_ROUNDS   intake gate settings (see gate/)
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
EXAMPLES_PATH = DATA_DIR / "examples.json"
FONT_DIR = DATA_DIR / "fonts"  # bundled Liberation Sans (SIL OFL, see fonts/LICENSE)
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "demo_cache"  # optional offline demo results, see app.load_cached


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=value lines). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


_load_dotenv()

PROVIDER = os.environ.get("CAD_PROVIDER", "anthropic").lower()
MODEL = os.environ.get("CAD_MODEL")  # None -> the provider's default in llm.PROVIDERS
GATE_MODEL = os.environ.get("CAD_GATE_MODEL")  # None -> the provider's default_gate_model, else MODEL
MAX_ATTEMPTS = int(os.environ.get("CAD_MAX_ATTEMPTS", "4"))
TOP_K = int(os.environ.get("CAD_TOP_K", "3"))
RAG_MIN_SIM = float(os.environ.get("CAD_RAG_MIN_SIM", "0"))
RETRIEVER = os.environ.get("CAD_RETRIEVER", "tfidf").lower()
GATE_PICKS = os.environ.get("CAD_GATE_PICKS", "0").lower() in ("1", "true", "yes", "on")
EMBED_MODEL = os.environ.get("CAD_EMBED_MODEL", "snowflake/snowflake-arctic-embed-s")  # best in eval/retrieval.py
# Downloaded model files (embedding models now; local LLMs later). Outside the repo so they are never committed.
MODEL_CACHE_DIR = Path(os.environ.get("CAD_MODEL_CACHE", Path.home() / ".cache" / "stl_agent"))
