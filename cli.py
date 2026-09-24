"""
Entry point: `ask "<claim>"`, `train --hours N`, `research --hours N`, `doctor`.

Build brief: docs/build-brief.md, step 11.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from config import load_settings
from engine import orchestrator
from engine.memory import knowledge_store, research_loop, session_log
from model_backend import trainer
from model_backend.brain import NoBrainError, load_brain
from model_backend.hardware_detect import detect_hardware

WIDTH = 88


def _wrap(text, indent: str = "  ", bullet: str = "") -> str:
    return textwrap.fill(str(text), WIDTH, initial_indent=indent + bullet,
                         subsequent_indent=indent + " " * len(bullet))


def _snippet(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _confidence(entry: dict) -> str:
    conf = entry["confidence"]
    if entry.get("confidence_before_cap"):
        conf += f", capped from {entry['confidence_before_cap']}"
    return conf


def render_persona(result: dict) -> list[str]:
    lines = [f"{result['persona'].upper()}  [{_confidence(result)}]  signal {result['signal']:+.2f}",
             _wrap(f'"{result["position"]}"')]
    for point in result["key_points"]:
        source = f" ({point['source']})" if point.get("source") else ""
        lines.append(_wrap(f"{point['delta']:+.2f} {_snippet(point['text'])}{source}", bullet="- "))
    return lines


def render_judge(judge: dict, title: str = "JUDGE") -> list[str]:
    lines = [f"{title}  [{_confidence(judge)}]  relied on: {', '.join(judge['relied_on'])}",
             _wrap(f"Verdict: {judge['verdict']}"),
             _wrap(f"Why: {judge['reasoning']}"),
             _wrap(f'In its own words: "{judge["in_its_own_words"]}"')]
    if judge.get("unresolved"):
        lines.append(_wrap(f"Unresolved: {judge['unresolved']}"))
    return lines


def render_result(result: dict) -> str:
    model, routing, fam = result["model"], result["routing"], result["familiarity"]
    investor = "Investor joins" if routing["is_money_idea"] else "no Investor"
    out = [
        _wrap(f"CLAIM: {result['claim']}", indent=""),
        f"Model: {model['description']}, {model['tokens_seen'] / 1e6:.1f}M tokens trained, "
        f"held-out loss {model['val_loss']:.2f}",
        _wrap(f"Router: {investor} - {routing['reason']}", indent=""),
        f"Familiarity: claim loss {fam['claim_loss']:.2f} vs. typical {fam['typical_loss']:.2f} "
        f"-> confidence ceiling {fam['cap']}",
        f"Knowledge base: {len(result['memory'])} relevant passage(s)",
        "",
    ]
    for persona in result["panel"].values():
        out += render_persona(persona) + [""]
    out += render_judge(result["judge"]) + [""]
    consistency = result.get("consistency")
    if consistency and not consistency["agreed"]:
        out += render_judge(consistency["rerun_judge"], title="JUDGE, REPEAT RUN") + [""]
        out.append(result["confidence_note"] + f" ({'; '.join(consistency['reasons'])}).")
    elif consistency:
        out.append("Repeat run (dropout on) agreed with the first.")
    out.append(f"OVERALL CONFIDENCE: {result['overall_confidence']}")
    return "\n".join(out)


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        result = orchestrator.evaluate(args.claim, recheck=False if args.no_recheck else None)
    except NoBrainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else render_result(result))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.data:
        print(f"Imported {trainer.import_texts(args.data, settings)} file(s) into "
              f"{trainer.corpus_dir(settings)}.")
    try:
        trainer.train(minutes=None if args.steps else args.hours * 60, steps=args.steps,
                      settings=settings)
    except trainer.TrainingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped; the checkpoint was saved.")
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    print(f"Researching and self-training for {args.hours:g} hour(s). Ctrl-C stops it; "
          "progress is saved as it goes.")
    try:
        summary = research_loop.run(args.hours)
    except KeyboardInterrupt:
        print("Stopped; everything collected so far is saved.")
        return 0
    print(f"Done: {len(summary['topics'])} topic(s), {summary['passages_stored']} new passages, "
          f"{summary['training_minutes']:.0f} min of training.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings()
    hw = detect_hardware(settings)
    size = hw["model_size"]
    print("Hardware")
    print(f"  RAM        {hw['ram_gb']} GB")
    print(f"  CPU cores  {hw['cpu_cores']}")
    print(f"  Tier       {hw['tier']} -> a new model would be {size['n_layer']} layers x "
          f"{size['d_model']} wide, {size['block_size']}-token context")

    files = trainer.corpus_files(settings)
    mb = sum(f.stat().st_size for f in files) / 1e6
    print("Corpus")
    print(f"  {len(files)} file(s), {mb:.1f} MB in {trainer.corpus_dir(settings)}")

    print("Model")
    try:
        brain = load_brain(settings)
    except NoBrainError:
        print("  none yet - run `python cli.py train --data <folder> --hours 1`")
    else:
        s = brain.stats
        print(f"  {brain.describe()}, vocabulary {brain.tokenizer.vocab_size}")
        print(f"  {s['steps']:,} steps, {s['tokens_seen'] / 1e6:.1f}M tokens seen, "
              f"{s['train_seconds'] / 3600:.1f} h trained, held-out loss {s['val_loss']:.3f}")

    print("Memory")
    print(f"  {knowledge_store.count(settings)} passage(s) in the knowledge base")
    print(f"  {len(session_log.sessions(settings))} past council session(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="council-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run a full council session on a claim.")
    ask.add_argument("claim", type=str)
    ask.add_argument("--no-recheck", action="store_true",
                     help="Skip the repeat run that checks the council agrees with itself.")
    ask.add_argument("--json", action="store_true", help="Print the raw result dict as JSON.")

    train = subparsers.add_parser("train", help="Train (or keep training) the model on the corpus.")
    train.add_argument("--data", help="Folder (or file) of .txt/.md text to add to the corpus first.")
    budget = train.add_mutually_exclusive_group()
    budget.add_argument("--hours", type=float, default=1.0,
                        help="How long to train, in hours (default 1; decimals work).")
    budget.add_argument("--steps", type=int, help="Train for this many steps instead of by time.")

    research = subparsers.add_parser("research", help="Research on Wikipedia and self-train.")
    research.add_argument("--hours", type=float, default=1.0,
                          help="How long to research and train, in hours (default 1).")

    subparsers.add_parser("doctor", help="Show hardware, model, and memory status.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="  ... %(message)s", stream=sys.stderr)
    return {"ask": cmd_ask, "train": cmd_train, "research": cmd_research,
            "doctor": cmd_doctor}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
