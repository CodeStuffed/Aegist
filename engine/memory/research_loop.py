"""
Autonomous research + self-training loop.

Repeats until time runs out:
  1. pick a topic: seed_topics from settings.yaml, then words that keep
     coming up in your past council sessions (a frequency count)
  2. fetch Wikipedia articles on it (plain search API - no AI involved)
  3. store their passages in the knowledge base, and the full text in the
     training corpus
  4. keep training the model on the grown corpus until the next fetch

Fetches are capped by research.max_topics_per_hour (to stay polite to
Wikipedia); training fills the time in between. State survives restarts
in data/research_state.json.

Build brief: docs/build-brief.md, step 9.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from config import data_dir, load_settings
from engine.memory import knowledge_store, session_log
from model_backend import trainer

USER_AGENT = "council-engine/0.1 (self-hosted research loop; personal use)"
HOUR = 3600
_STOP_SECTIONS = re.compile(r"^==+\s*(References|See also|External links|Notes|Further reading|"
                            r"Bibliography|Sources|Citations)\s*==+\s*$", re.MULTILINE | re.IGNORECASE)
_HEADING = re.compile(r"^==+.*==+\s*$", re.MULTILINE)


def _state_path(settings):
    return data_dir(settings) / "research_state.json"


def load_state(settings) -> dict:
    path = _state_path(settings)
    state = json.loads(path.read_text()) if path.exists() else {}
    return {"fetches": state.get("fetches", []), "topics": state.get("topics", {})}


def save_state(state: dict, settings) -> None:
    _state_path(settings).write_text(json.dumps(state, indent=1))


def seconds_until_fetch_allowed(state: dict, settings, now: float) -> float:
    """0 if under max_topics_per_hour for the trailing hour, else the wait."""
    recent = sorted(t for t in state["fetches"] if now - t < HOUR)
    state["fetches"] = recent
    cap = settings["research"]["max_topics_per_hour"]
    return 0.0 if len(recent) < cap else recent[len(recent) - cap] + HOUR - now


def due_topics(state: dict, settings, now: float) -> list[str]:
    """Seed topics first, then recurring session keywords (most frequent
    first), minus anything fetched within refresh_hours."""
    cfg = settings["research"]
    frequent = [k for k, _ in session_log.keyword_counts(settings).most_common()]
    candidates = list(dict.fromkeys([t.strip().lower() for t in cfg["seed_topics"]] + frequent))
    fresh = cfg["refresh_hours"] * HOUR
    return [t for t in candidates if now - state["topics"].get(t, -fresh) >= fresh]


def fetch_wikipedia(topic: str, settings) -> list[dict]:
    """Search Wikipedia for `topic`; return [{"title", "url", "text"}] (plain text)."""
    cfg = settings["research"]
    api = cfg["wikipedia_api"]
    headers = {"User-Agent": USER_AGENT}
    found = requests.get(api, headers=headers, timeout=30, params={
        "action": "query", "list": "search", "srsearch": topic, "srlimit": cfg["articles_per_topic"],
        "format": "json", "formatversion": 2}).json()
    articles = []
    for hit in found.get("query", {}).get("search", []):
        page = requests.get(api, headers=headers, timeout=30, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
            "titles": hit["title"], "format": "json", "formatversion": 2}).json()
        for p in page.get("query", {}).get("pages", []):
            if p.get("extract"):
                url = api.replace("/w/api.php", "/wiki/") + quote(p["title"].replace(" ", "_"))
                articles.append({"title": p["title"], "url": url, "text": p["extract"]})
    return articles


def split_passages(text: str, min_chars: int) -> list[str]:
    """Article body -> paragraphs, minus headings and reference sections."""
    stop = _STOP_SECTIONS.search(text)
    body = _HEADING.sub("", text[: stop.start()] if stop else text)
    return [" ".join(p.split()) for p in body.split("\n") if len(p.strip()) >= min_chars]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80] or "article"


def store_articles(topic: str, articles: list[dict], settings) -> int:
    """Passages -> knowledge base; full text -> training corpus. Returns
    how many new passages were stored."""
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    corpus = trainer.corpus_dir(settings) / "research"
    corpus.mkdir(parents=True, exist_ok=True)
    stored = 0
    for article in articles:
        passages = split_passages(article["text"], settings["research"]["min_passage_chars"])
        for passage in passages:
            meta = {"topic": topic, "title": article["title"], "url": article["url"],
                    "retrieved_at": retrieved_at, "source": "wikipedia"}
            if knowledge_store.add(passage, meta, settings=settings):
                stored += 1
        if passages:
            (corpus / f"{_slug(article['title'])}.txt").write_text("\n\n".join(passages), encoding="utf-8")
    return stored


def run(hours: float, *, settings: dict | None = None, log=print, clock=time.time,
        sleep=time.sleep, fetch=fetch_wikipedia, train=trainer.train) -> dict:
    """Research and self-train for `hours` (Ctrl-C stops it; progress is saved)."""
    settings = settings if settings is not None else load_settings()
    cfg = settings["research"]
    deadline = clock() + hours * HOUR
    state = load_state(settings)
    summary = {"topics": [], "passages_stored": 0, "training_minutes": 0.0}

    while clock() < deadline:
        now = clock()
        topics = due_topics(state, settings, now)
        wait = seconds_until_fetch_allowed(state, settings, now)
        if topics and wait == 0:
            topic = topics[0]
            state["fetches"].append(now)
            state["topics"][topic] = now
            save_state(state, settings)
            try:
                articles = fetch(topic, settings)
            except (requests.RequestException, ValueError) as e:
                # Still counts toward the hourly cap, but the topic stays due.
                del state["topics"][topic]
                save_state(state, settings)
                log(f"Couldn't fetch '{topic}' from Wikipedia: {e.__class__.__name__}: {e}")
            else:
                stored = store_articles(topic, articles, settings)
                titles = ", ".join(a["title"] for a in articles) or "nothing found"
                log(f"Researched '{topic}': {stored} new passages ({titles}).")
                summary["topics"].append(topic)
                summary["passages_stored"] += stored

        # Self-train until the next fetch is due.
        burst = min(cfg["train_minutes_per_topic"] * 60, deadline - clock())
        if burst <= 0:
            break
        started = clock()
        try:
            train(minutes=burst / 60, settings=settings, log=log)
            summary["training_minutes"] += (clock() - started) / 60
        except trainer.TrainingError as e:
            log(f"Not training yet: {e}")
            # Nothing to train on: wait for the next fetch slot instead.
            now = clock()
            idle = seconds_until_fetch_allowed(state, settings, now) if due_topics(state, settings, now) else burst
            sleep(min(max(idle, 1.0), burst, max(deadline - now, 0)))
    return summary
