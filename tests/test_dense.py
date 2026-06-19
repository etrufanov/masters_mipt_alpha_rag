import numpy as np
import pandas as pd
from rag.retriever.dense import DenseRetriever


class FakeEmbedder:
    def encode_queries(self, texts):
        # query aligns with the second chunk's vector
        return np.array([[0.0, 1.0]], dtype="float32")


def _store():
    chunks = pd.DataFrame([
        {"chunk_id": "1:0", "web_id": 1, "url": "u", "title": "t0", "text": "alpha"},
        {"chunk_id": "2:0", "web_id": 2, "url": "u", "title": "t1", "text": "beta"},
    ])
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    return chunks, embs


def test_returns_topk_sorted_by_score():
    chunks, embs = _store()
    r = DenseRetriever(chunks, embs, FakeEmbedder())
    out = r.search("q", k=2)
    assert out[0].chunk_id == "2:0"          # best match first
    assert out[0].score > out[1].score
    assert out[0].text == "beta"


def test_k_limits_results():
    chunks, embs = _store()
    r = DenseRetriever(chunks, embs, FakeEmbedder())
    assert len(r.search("q", k=1)) == 1
