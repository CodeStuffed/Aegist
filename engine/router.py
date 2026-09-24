"""
Decides whether the Investor persona should run, by checking whether the
claim is about making money. Plain keyword matching against
router.money_keywords in settings.yaml - transparent and easy to extend.

Build brief: docs/build-brief.md, step 4.
"""

from __future__ import annotations

from config import load_settings
from engine.text import keywords, words


def route(claim: str, settings: dict | None = None) -> dict:
    """Return {"is_money_idea": bool, "reason": str, "keywords": [str]}.

    `keywords` are the claim's distinctive words; the research loop counts
    them across sessions to find recurring subjects worth reading up on.
    """
    settings = settings if settings is not None else load_settings()
    lexicon = {k.lower() for k in settings["router"]["money_keywords"]}
    claim_words = set(words(claim))
    hits = sorted(w for w in claim_words if w in lexicon)
    if "$" in lexicon and "$" in claim:
        hits.append("$")
    reason = (f"Mentions {', '.join(hits)}." if hits
              else "No money-related words, so no Investor.")
    return {"is_money_idea": bool(hits), "reason": reason, "keywords": keywords(claim)}
