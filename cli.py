"""
Entry point: `ask "<claim>"`, `doctor`, `research --hours N`.

Build brief: docs/build-brief.md, step 11.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

import model_backend
from config import load_settings
from engine import orchestrator
from model_backend import cloud_runner, local_runner
from model_backend.cloud_runner import CloudBackendError
from model_backend.hardware_detect import detect_hardware
from model_backend.local_runner import LocalBackendError

WIDTH = 88
BACKEND_ERRORS = (CloudBackendError, LocalBackendError, model_backend.BackendUnavailable)


def _wrap(text, indent: str = "  ", bullet: str = "") -> str:
    return textwrap.fill(str(text), WIDTH, initial_indent=indent + bullet,
                         subsequent_indent=indent + " " * len(bullet))


def render_persona(result: dict) -> list[str]:
    lines = [f"{result['persona'].upper()}  [{result.get('confidence') or 'no confidence given'}]"]
    if "error" in result:
        lines.append(_wrap(f"({result['error']}) {result.get('raw', '')}"))
        return lines
    lines.append(_wrap(result.get("position", "")))
    for point in result.get("key_points") or []:
        lines.append(_wrap(point, bullet="- "))
    if result.get("what_would_change_my_mind"):
        lines.append(_wrap(f"Would change my mind: {result['what_would_change_my_mind']}"))
    return lines


def render_judge(judge: dict, title: str = "JUDGE") -> list[str]:
    conf = judge.get("confidence")
    if judge.get("confidence_before_cap"):
        conf += f" (said {judge['confidence_before_cap']}; capped by the weakest input it relied on)"
    lines = [f"{title}  [{conf}]"]
    if "error" in judge:
        lines.append(_wrap(f"({judge['error']}) {judge.get('raw', '')}"))
        return lines
    lines.append(_wrap(f"Verdict: {judge.get('verdict', '')}"))
    lines.append(_wrap(f"Reasoning: {judge.get('reasoning', '')}"))
    if judge.get("unresolved"):
        lines.append(_wrap(f"Unresolved: {judge['unresolved']}"))
    return lines


def render_result(result: dict) -> str:
    models = result["models"]
    routing = result["routing"]
    investor = "Investor joins" if routing["is_money_idea"] else "no Investor"
    out = [
        _wrap(f"CLAIM: {result['claim']}", indent=""),
        f"Backend: {result['backend']} (judge {models['judge']}, panel {models['panel']})",
        _wrap(f"Router: {investor} - {routing['reason']}", indent=""),
        "",
    ]
    for persona in result["panel"].values():
        out += render_persona(persona) + [""]
    out += render_judge(result["judge"]) + [""]
    out.append(f"OVERALL CONFIDENCE: {result['overall_confidence']}")
    return "\n".join(out)


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        result = orchestrator.evaluate(args.claim, backend=args.backend)
    except BACKEND_ERRORS as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else render_result(result))
    return 0


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
    ask.add_argument("--backend", choices=model_backend.CHOICES, default=None,
                     help="Override backend.prefer from settings.yaml.")
    ask.add_argument("--json", action="store_true", help="Print the raw result dict as JSON.")

    doctor = subparsers.add_parser("doctor", help="Show detected hardware and active backend.")
    doctor.add_argument("--backend", choices=model_backend.CHOICES, default=None,
                        help="Override backend.prefer from settings.yaml.")

    research = subparsers.add_parser("research", help="Run the autonomous research loop.")
    research.add_argument("--hours", type=float, default=1.0)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="  ... %(message)s", stream=sys.stderr)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    raise NotImplementedError(f"'{args.command}' is not implemented yet - see docs/build-brief.md")


if __name__ == "__main__":
    raise SystemExit(main())
