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
    CAD_RPM             max LLM requests per minute per model (Gemini default 14; free tier allows 15)
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
