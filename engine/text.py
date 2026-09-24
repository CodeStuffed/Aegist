"""Word-level helpers shared by the router and the knowledge store."""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have haven't
having he her here hers herself him himself his how i if in into is isn't it it's its itself
just let's may me might more most much must mustn't my myself no nor not now of off on once
only or other ought our ours ourselves out over own same shall she should shouldn't so some
such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up upon us very was wasn't we
we'd we'll we're we've were weren't what what's when where which while who whom why will
with won't would wouldn't you your yours yourself yourselves one two also get got make made
really thing things way ways lot lots every going
""".split())


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def content_words(text: str) -> list[str]:
    """Lowercase words minus stopwords and very short tokens, in order."""
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2]


def keywords(text: str, limit: int = 4) -> list[str]:
    """The claim's most distinctive words: longest content words first, deduped."""
    unique = list(dict.fromkeys(w for w in content_words(text) if len(w) >= 4 and not w.isdigit()))
    return sorted(unique, key=len, reverse=True)[:limit]
