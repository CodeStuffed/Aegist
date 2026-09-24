"""
Entry point: `ask "<claim>"`, `doctor`, `research --hours N`.

Build brief: docs/build-brief.md, step 11.
"""

from __future__ import annotations

import argparse

from config import load_settings
from model_backend.hardware_detect import detect_hardware


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings()
    hw = detect_hardware(settings)
    print("Hardware")
    print(f"  RAM        {hw['ram_gb']} GB")
    print(f"  VRAM       {hw['vram_gb']} GB  ({hw['vram_source']})")
    print(f"  CPU cores  {hw['cpu_cores']}")
    print(f"  Tier       {hw['tier']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="council-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run a full council session on a claim.")
    ask.add_argument("claim", type=str)

    subparsers.add_parser("doctor", help="Show detected hardware and active backend.")

    research = subparsers.add_parser("research", help="Run the autonomous research loop.")
    research.add_argument("--hours", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "doctor":
        return cmd_doctor(args)
    raise NotImplementedError(f"'{args.command}' is not implemented yet - see docs/build-brief.md")


if __name__ == "__main__":
    raise SystemExit(main())
