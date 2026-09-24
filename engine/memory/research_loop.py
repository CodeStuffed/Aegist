"""
Autonomous research loop: uses the Anthropic API's built-in web_search
tool to research topics and store sourced findings in the knowledge
store. Rate-limited by research.max_calls_per_hour in settings.yaml.

Build brief: docs/build-brief.md, step 9.
"""

from __future__ import annotations


def run(hours: float) -> None:
    raise NotImplementedError
