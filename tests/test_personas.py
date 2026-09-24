"""Step 3: persona prompts are exactly the text in the build brief."""

from __future__ import annotations

import re

import pytest

from config import PERSONA_DIR, ROOT

BRIEF = (ROOT / "docs" / "build-brief.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["believer", "skeptic", "investor", "judge"])
def test_persona_matches_brief_verbatim(name):
    m = re.search(rf"--- {name}\.md ---\n(.*?)\n(?=\n|--- )", BRIEF, re.S)
    assert m, f"{name}.md block not found in docs/build-brief.md"
    assert (PERSONA_DIR / f"{name}.md").read_text(encoding="utf-8") == m.group(1).rstrip() + "\n"
