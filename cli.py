"""
Entry point: `ask "<claim>"`, `doctor`, `research --hours N`.

Build brief: docs/build-brief.md, step 11.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="council-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run a full council session on a claim.")
    ask.add_argument("claim", type=str)

    subparsers.add_parser("doctor", help="Show detected hardware and active backend.")

    research = subparsers.add_parser("research", help="Run the autonomous research loop.")
    research.add_argument("--hours", type=float, default=1.0)

    args = parser.parse_args()
    raise NotImplementedError(f"'{args.command}' is not implemented yet - see docs/build-brief.md")


if __name__ == "__main__":
    main()
