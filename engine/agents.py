"""
Agent: one persona on the council, backed by the from-scratch model.

A persona (config/personas/<name>.yaml) is a lead-in the model continues
to write its position, plus probe phrases it is scored on. The persona's
**signal** is pointwise mutual information: how much more likely the model
finds its probes right after the claim than after a neutral lead-in.

Build brief: docs/build-brief.md, step 5.
"""

from __future__ import annotations

import numpy as np
import yaml

from config import PERSONA_DIR, load_settings

CONFIDENCE_LEVELS = ("Low", "Medium", "High")  # ordered weakest -> strongest

_baselines: dict[tuple, float] = {}


def min_confidence(*levels: str | None) -> str:
    """The weakest of the given levels; missing counts as Low."""
    ranks = [CONFIDENCE_LEVELS.index(lv) if lv in CONFIDENCE_LEVELS else 0 for lv in levels]
    return CONFIDENCE_LEVELS[min(ranks)]


def confidence_from_signal(signal: float, thresholds: dict) -> str:
    if signal >= thresholds["high"]:
        return "High"
    if signal >= thresholds["medium"]:
        return "Medium"
    return "Low"


def as_sentence(claim: str) -> str:
    claim = " ".join(claim.split())
    return claim if claim.endswith((".", "!", "?")) else claim + "."


def build_prompt(claim: str, passages: list[dict]) -> str:
    """Evidence first, claim last. Passages arrive most relevant first; they
    go in reverse so the best one sits next to the claim, since a long
    prompt is cut from the left to fit the model's context."""
    parts = [p["text"] for p in reversed(passages)] + [as_sentence(claim)]
    return "\n\n".join(parts)


def load_persona(name: str) -> dict:
    path = PERSONA_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PERSONA_DIR.glob("*.yaml"))
        raise ValueError(f"No persona {name!r}; available: {available}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Agent:
    def __init__(self, persona_name: str, brain, settings: dict | None = None):
        self.persona_name = persona_name
        self.brain = brain
        self.settings = settings if settings is not None else load_settings()
        self.persona = load_persona(persona_name)

    def _pmi(self, prompt: str, probes: list[str], rng) -> float:
        neutral = self.settings["council"]["neutral_prompt"]
        total = 0.0
        for probe in probes:
            key = (id(self.brain), self.brain.stats.get("updated_at"), neutral, probe)
            if key not in _baselines:
                _baselines[key] = self.brain.continuation_logprob(neutral, probe)
            total += self.brain.continuation_logprob(prompt, probe, rng) - _baselines[key]
        return total / len(probes)

    def signal(self, prompt: str, rng: np.random.Generator | None = None) -> float:
        """PMI of this persona's probes after `prompt` (minus counter-probes)."""
        value = self._pmi(prompt, self.persona["probes"], rng)
        if self.persona.get("counter_probes"):
            value -= self._pmi(prompt, self.persona["counter_probes"], rng)
        return value

    def speak(self, prompt: str, rng: np.random.Generator | None = None) -> str:
        """The model continues the prompt from this persona's lead-in."""
        lead_in = self.persona["lead_in"]
        gen = self.settings["council"]["generate"]
        text = self.brain.generate(f"{prompt} {lead_in}", max_new_tokens=gen["max_new_tokens"],
                                   temperature=gen["temperature"], top_k=gen["top_k"], rng=rng)
        return f"{lead_in} {text}".strip()

    def respond(self, claim: str, passages: list[dict] | None = None, *,
                familiarity_cap: str = "High", rng: np.random.Generator | None = None,
                gen_rng: np.random.Generator | None = None) -> dict:
        """Score the claim for this persona and write its position.

        `rng` switches dropout on while scoring (the uncertainty re-run);
        `gen_rng` drives sampling of the written position.
        """
        passages = passages or []
        prompt = build_prompt(claim, passages)
        signal = self.signal(prompt, rng)

        # Which stored passages push the model toward this persona's side?
        key_points = []
        if passages:
            without = self.signal(build_prompt(claim, []), rng)
            for p in passages:
                delta = self.signal(build_prompt(claim, [p]), rng) - without
                if delta > 0:
                    meta = p.get("metadata", {})
                    key_points.append({"delta": round(delta, 3), "text": p["text"],
                                       "source": meta.get("title") or meta.get("source", ""),
                                       "url": meta.get("url", "")})
            key_points.sort(key=lambda k: k["delta"], reverse=True)

        own = confidence_from_signal(signal, self.settings["council"]["confidence_thresholds"])
        confidence = min_confidence(own, familiarity_cap)
        result = {
            "persona": self.persona_name,
            "role": self.persona.get("role", ""),
            "position": self.speak(prompt, gen_rng),
            "key_points": key_points,
            "signal": round(signal, 3),
            "confidence": confidence,
        }
        if confidence != own:
            result["confidence_before_cap"] = own
        return result
