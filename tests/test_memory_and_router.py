"""BM25 knowledge store, session log, and the keyword router."""

from __future__ import annotations

from engine.memory import knowledge_store, session_log
from engine.memory.knowledge_store import KnowledgeStore
from engine.router import route


def test_add_dedupes_and_persists(settings):
    assert knowledge_store.add("Prisms split white light into colours.", {"title": "Prism"}, settings)
    assert knowledge_store.add("Prisms  split white light\ninto colours.", {}, settings) is None
    knowledge_store._stores.clear()  # force a reload from disk
    assert knowledge_store.count(settings) == 1


def test_bm25_ranks_the_relevant_passage_first(tmp_path):
    kb = KnowledgeStore(tmp_path / "kb.jsonl")
    kb.add("Usage-based pricing charges customers per unit consumed.", {"title": "Pricing"})
    kb.add("Glass refracts light because light slows down in it.", {"title": "Optics"})
    kb.add("Subscription pricing charges a flat monthly fee.", {"title": "Subscriptions"})
    hits = kb.search("should we move to usage-based pricing?", k=2)
    assert [h["metadata"]["title"] for h in hits] == ["Pricing", "Subscriptions"]
    top = kb.search("refracts light glass", k=5, min_relevance=0.3)
    assert [h["metadata"]["title"] for h in top] == ["Optics"] and 0 < top[0]["relevance"] <= 1
    assert kb.search("kangaroo", k=5) == []


def test_search_on_empty_store(settings):
    assert knowledge_store.search("anything", settings=settings) == []


def test_router(settings):
    money = route("We should charge $9/month for a pro tier", settings)
    assert money["is_money_idea"] and "charge" in money["reason"]
    plain = route("The Great Wall of China is visible from orbit", settings)
    assert not plain["is_money_idea"]
    assert "visible" in plain["keywords"] and "the" not in plain["keywords"]


def test_session_log_counts_keywords(settings):
    for claim in ["pricing matters", "pricing is hard", "orbit"]:
        session_log.record({"timestamp": "t", "claim": claim, "routing": {"keywords": claim.split()[:1]},
                            "judge": {"stance": "yes"}, "overall_confidence": "Low"}, settings)
    assert session_log.keyword_counts(settings).most_common(1) == [("pricing", 2)]
