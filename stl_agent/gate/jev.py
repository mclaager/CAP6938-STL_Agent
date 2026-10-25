"""Gate backed by TypeSafe Jev (not implemented yet; see issue #12)."""

from __future__ import annotations

from .base import Decision


class JevGate:
    """Placeholder for a TypeSafe Jev backend (calibrated classifier, limited early access).

    Plan: send the transcript as `state` with a `choice` question for scope and `noul` questions
    such as "Must the part fit an object whose dimensions are not given?", then gate on the returned
    confidence. Jev cannot write questions or a spec, so CLARIFY still needs an LLM call.
    """

    def decide(self, conversation: list[dict], rounds_asked: int = 0) -> Decision:
        raise NotImplementedError("Jev gate is not implemented yet; set gate = 'llm' or 'off' in settings.toml.")
