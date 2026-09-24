"""The research + self-training loop, against a fake Wikipedia and a fake clock."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from engine.memory import knowledge_store, research_loop, session_log
from model_backend import trainer

ARTICLE = ("Pricing is the process of setting a price. " * 6 + "\n"
           + "Value-based pricing sets prices by what customers will pay. " * 5 + "\n"
           + "== History ==\n"
           + "Short line.\n"
           + "Early merchants priced goods by haggling in open markets over many centuries. " * 3 + "\n"
           + "== References ==\n"
           + "Smith, A. (1776). The Wealth of Nations. A very long citation line that should be dropped.")


class FakeWikipedia(BaseHTTPRequestHandler):
    requests_seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        FakeWikipedia.requests_seen.append(q)
        if q.get("list") == "search":
            body = {"query": {"search": [{"title": "Pricing"}, {"title": "Price discrimination"}]}}
        else:
            body = {"query": {"pages": [{"title": q["titles"], "extract": ARTICLE.replace("Pricing", q["titles"])}]}}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def wikipedia(settings):
    FakeWikipedia.requests_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeWikipedia)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    settings["research"]["wikipedia_api"] = f"http://127.0.0.1:{server.server_address[1]}/w/api.php"
    yield server
    server.shutdown()


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_split_passages_drops_headings_short_lines_and_references():
    passages = research_loop.split_passages(ARTICLE, 150)
    assert len(passages) == 3
    assert not any("==" in p or "Wealth of Nations" in p or p == "Short line." for p in passages)


def test_fetch_wikipedia_uses_plain_search_api(settings, wikipedia):
    articles = research_loop.fetch_wikipedia("pricing strategy", settings)
    assert [a["title"] for a in articles] == ["Pricing", "Price discrimination"]
    assert articles[1]["url"].endswith("/wiki/Price_discrimination")
    search, extract = FakeWikipedia.requests_seen[:2]
    assert search["srsearch"] == "pricing strategy" and search["srlimit"] == "2"
    assert extract["prop"] == "extracts" and extract["explaintext"] == "1"


def test_store_articles_feeds_knowledge_base_and_corpus(settings, wikipedia):
    articles = research_loop.fetch_wikipedia("pricing", settings)
    # 3 passages from the first article; the second repeats two of them word for word
    assert research_loop.store_articles("pricing", articles, settings) == 4
    assert research_loop.store_articles("pricing", articles, settings) == 0
    hit = knowledge_store.search("value-based pricing customers", k=1, settings=settings)[0]
    assert hit["metadata"]["url"].startswith("http") and hit["metadata"]["topic"] == "pricing"
    assert len(list((trainer.corpus_dir(settings) / "research").glob("*.txt"))) == 2


def test_due_topics_seeds_then_frequent_session_words(settings):
    settings["research"]["seed_topics"] = ["Pricing Strategy"]
    for kw in (["churn"], ["churn", "onboarding"]):
        session_log.record({"timestamp": "t", "claim": "c", "routing": {"keywords": kw},
                            "judge": {"stance": "yes"}, "overall_confidence": "Low"}, settings)
    state = {"fetches": [], "topics": {}}
    assert research_loop.due_topics(state, settings, 0) == ["pricing strategy", "churn", "onboarding"]
    state["topics"]["churn"] = 0
    assert "churn" not in research_loop.due_topics(state, settings, 3600)
    assert "churn" in research_loop.due_topics(state, settings, 169 * 3600)


def test_rate_limit_per_trailing_hour(settings):
    settings["research"]["max_topics_per_hour"] = 2
    state = {"fetches": [0.0, 600.0], "topics": {}}
    assert research_loop.seconds_until_fetch_allowed(state, settings, 1200) == 2400
    assert research_loop.seconds_until_fetch_allowed(state, settings, 3601) == 0


def test_run_alternates_research_and_training(settings, wikipedia):
    settings["research"].update(seed_topics=["pricing", "markets", "profit"], max_topics_per_hour=2,
                                train_minutes_per_topic=5)
    clock, trained = FakeClock(), []

    def fake_train(minutes, settings, log):
        trained.append(minutes)
        clock.now += minutes * 60

    summary = research_loop.run(0.5, settings=settings, log=lambda *_: None, clock=clock,
                                sleep=clock.sleep, train=fake_train)
    # 30 minutes: fetch, train 5, fetch, train 5, then the hourly cap blocks fetches
    assert summary["topics"] == ["pricing", "markets"]
    assert sum(trained) == pytest.approx(30)
    assert knowledge_store.count(settings) > 0
    state = research_loop.load_state(settings)  # survives a restart
    assert set(state["topics"]) == {"pricing", "markets"} and len(state["fetches"]) == 2


def test_run_waits_when_nothing_to_train_on(settings):
    settings["research"]["seed_topics"] = []
    clock = FakeClock()
    summary = research_loop.run(0.25, settings=settings, log=lambda *_: None, clock=clock,
                                sleep=clock.sleep, fetch=lambda *a: [])
    assert summary == {"topics": [], "passages_stored": 0, "training_minutes": 0.0}
    assert clock.now == pytest.approx(1_000_000 + 900)


def test_failed_fetch_keeps_topic_due_but_counts_toward_cap(settings):
    import requests
    settings["research"].update(seed_topics=["pricing"], max_topics_per_hour=3)
    clock = FakeClock()

    def offline(topic, settings):
        raise requests.ConnectionError("no network")

    research_loop.run(0.1, settings=settings, log=lambda *_: None, clock=clock, sleep=clock.sleep,
                      fetch=offline)
    state = research_loop.load_state(settings)
    assert state["topics"] == {} and len(state["fetches"]) == 3  # retried, but only up to the cap
