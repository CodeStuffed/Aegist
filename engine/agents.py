"""
Agent wrapper: loads a persona's system prompt from config/personas/,
calls a backend (local or cloud), and parses the persona's structured
JSON reply.

Build brief: docs/build-brief.md, step 5.
"""

from __future__ import annotations

from config import PERSONA_DIR
from engine.llm_json import call_json

CONFIDENCE_LEVELS = ("Low", "Medium", "High")  # ordered weakest -> strongest


def normalize_confidence(value) -> str | None:
    """"high" / " HIGH " -> "High"; anything unrecognized -> None."""
    if not isinstance(value, str):
        return None
    value = value.strip().capitalize()
    return value if value in CONFIDENCE_LEVELS else None


class Agent:
    def __init__(self, persona_name: str, backend):
        self.persona_name = persona_name
        self.backend = backend
        path = PERSONA_DIR / f"{persona_name}.md"
        if not path.exists():
            available = sorted(p.stem for p in PERSONA_DIR.glob("*.md"))
            raise ValueError(f"No persona {persona_name!r}; available: {available}")
        self.system_prompt = path.read_text(encoding="utf-8")
        # The Judge gets the stronger model; everyone else is the panel.
        self.role = "judge" if persona_name == "judge" else "panel"

    def build_prompt(self, claim: str, context: str = "") -> str:
        parts = [f"CLAIM:\n{claim.strip()}"]
        if context.strip():
            parts.append(context.strip())
        return "\n\n".join(parts)

    def respond(self, claim: str, context: str = "", *, temperature: float | None = None) -> dict:
        """Return the persona's parsed JSON reply.

        Always includes "persona", "backend", "model", and "confidence"
        (normalized to High/Medium/Low, or None). If the reply still isn't
        JSON after one retry, "error" and the "raw" text are returned instead
        of the persona's fields.
        """
        parsed, meta = call_json(self.backend, self.system_prompt,
                                 self.build_prompt(claim, context),
                                 role=self.role, temperature=temperature)
        if parsed is None:
            result = {"error": "Reply wasn't valid JSON, even after a retry.", "raw": meta["raw"]}
        else:
            result = dict(parsed)
        result.update(persona=self.persona_name, backend=meta["backend"], model=meta["model"],
                      attempts=meta["attempts"])
        result["confidence"] = normalize_confidence(result.get("confidence"))
        return result
