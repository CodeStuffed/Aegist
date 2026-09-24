"""
Entry point: `ask "<claim>"`, `doctor`, `research --hours N`.

Build brief: docs/build-brief.md, step 11.
"""

from __future__ import annotations

import argparse

import model_backend
from config import load_settings
from model_backend import cloud_runner, local_runner
from model_backend.hardware_detect import detect_hardware


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings()
    hw = detect_hardware(settings)
    print("Hardware")
    print(f"  RAM        {hw['ram_gb']} GB")
    print(f"  VRAM       {hw['vram_gb']} GB  ({hw['vram_source']})")
    print(f"  CPU cores  {hw['cpu_cores']}")
    print(f"  Tier       {hw['tier']}")

    ok, why = cloud_runner.is_available(settings)
    print("Cloud (Anthropic API)")
    print(f"  Status     {'available' if ok else 'unavailable'} - {why}")
    print(f"  Judge      {cloud_runner.model_for('judge', settings)}")
    print(f"  Panel      {cloud_runner.model_for('panel', settings)}")

    local = local_runner.status(settings)
    print("Local (Ollama)")
    print(f"  Host       {local['host']} - {'reachable' if local['reachable'] else 'not reachable'}")
    pulled = "pulled" if local["model_pulled"] else f"not pulled (ollama pull {local['model']})"
    print(f"  Model      {local['model']} - {pulled}")

    print("Active")
    try:
        backend = model_backend.select_backend(args.backend, settings)
    except model_backend.BackendUnavailable as e:
        print(f"  Backend    none - {e}")
        return 1
    print(f"  Backend    {backend.NAME}")
    print(f"  Judge      {backend.model_for('judge', settings)}")
    print(f"  Panel      {backend.model_for('panel', settings)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="council-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run a full council session on a claim.")
    ask.add_argument("claim", type=str)

    doctor = subparsers.add_parser("doctor", help="Show detected hardware and active backend.")
    doctor.add_argument("--backend", choices=model_backend.CHOICES, default=None,
                        help="Override backend.prefer from settings.yaml.")

    research = subparsers.add_parser("research", help="Run the autonomous research loop.")
    research.add_argument("--hours", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "doctor":
        return cmd_doctor(args)
    raise NotImplementedError(f"'{args.command}' is not implemented yet - see docs/build-brief.md")


if __name__ == "__main__":
    raise SystemExit(main())
