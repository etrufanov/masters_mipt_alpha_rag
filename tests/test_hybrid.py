from rag.retriever.hybrid import rrf_fuse, HybridRetriever
from rag.schema import Retrieved

def _r(cid, score):
    return Retrieved(chunk_id=cid, web_id=int(cid.split(":")[0]), url="u", title="t", text=cid, score=score)

def test_rrf_fuse_rewards_agreement():
    dense = [_r("1:0", 0.9), _r("2:0", 0.8), _r("3:0", 0.1)]
    bm25  = [_r("2:0", 5.0), _r("1:0", 4.0), _r("4:0", 1.0)]
    fused = rrf_fuse([dense, bm25], k=60)
    # 1:0 and 2:0 appear high in both -> ranked above 3:0/4:0
    top2 = {c.chunk_id for c in fused[:2]}
    assert top2 == {"1:0", "2:0"}

class FakeReranker:
    # reverse the candidate order so we can assert rerank is applied
    def rerank(self, query, cands, top_n):
        return list(reversed(cands))[:top_n]

class FakeRetr:
    def __init__(self, items): self.items = items
    def search(self, query, k): return self.items[:k]

def test_hybrid_applies_rerank_after_fusion():
    dense = FakeRetr([_r("1:0", 0.9), _r("2:0", 0.5)])
    bm25 = FakeRetr([_r("2:0", 3.0), _r("1:0", 1.0)])
    h = HybridRetriever(dense, bm25, FakeReranker(), k_dense=2, k_bm25=2, rrf_k=60)
    out = h.search("q", k=2)
    assert len(out) == 2
    assert out[0].chunk_id == "2:0"   # reversed by FakeReranker
