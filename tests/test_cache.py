import pytest
from rag.schema import Retrieved
from rag.retriever.cache import cached_search


class CountingRetriever:
    """Records how many times search() is called and returns a deterministic context."""
    def __init__(self):
        self.calls = 0

    def search(self, q, k):
        self.calls += 1
        return [Retrieved(chunk_id=f"{q}:0", web_id=1, url="u", title="t",
                          text=f"ctx for {q}", score=0.5)]


def _path(tmp_path):
    return tmp_path / "retrieval_cache.json"


def test_fills_then_reuses_without_rebuilding(tmp_path):
    r = CountingRetriever()
    ctxs = cached_search(lambda: r, ["a", "b"], ["qa", "qb"], k=1,
                         cache_path=_path(tmp_path), sig="v1", progress=False)
    assert r.calls == 2
    assert [c[0].text for c in ctxs] == ["ctx for qa", "ctx for qb"]

    # full cache hit: factory must NOT be invoked at all
    def boom():
        raise AssertionError("retriever should not be built on a full cache hit")
    ctxs2 = cached_search(boom, ["a", "b"], ["qa", "qb"], k=1,
                          cache_path=_path(tmp_path), sig="v1", progress=False)
    assert [c[0].text for c in ctxs2] == ["ctx for qa", "ctx for qb"]


def test_incremental_fill(tmp_path):
    r1 = CountingRetriever()
    cached_search(lambda: r1, ["a"], ["qa"], k=1,
                  cache_path=_path(tmp_path), sig="v1", progress=False)
    assert r1.calls == 1

    r2 = CountingRetriever()
    ctxs = cached_search(lambda: r2, ["a", "b"], ["qa", "qb"], k=1,
                         cache_path=_path(tmp_path), sig="v1", progress=False)
    assert r2.calls == 1  # only the new q_id "b" recomputed
    assert [c[0].text for c in ctxs] == ["ctx for qa", "ctx for qb"]


def test_signature_change_invalidates(tmp_path):
    r1 = CountingRetriever()
    cached_search(lambda: r1, ["a"], ["qa"], k=1,
                  cache_path=_path(tmp_path), sig="v1", progress=False)

    r2 = CountingRetriever()
    cached_search(lambda: r2, ["a"], ["qa"], k=1,
                  cache_path=_path(tmp_path), sig="v2", progress=False)
    assert r2.calls == 1  # stale signature -> recompute
