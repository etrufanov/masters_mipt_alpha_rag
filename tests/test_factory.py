import numpy as np, pandas as pd
from rag.config import RetrievalConfig
from rag.retriever.factory import build_retriever
from rag.retriever.dense import DenseRetriever

class FakeEmbedder:
    def encode_queries(self, texts):
        return np.array([[1.0, 0.0]], dtype="float32")

def _chunks():
    return pd.DataFrame([{"chunk_id": "1:0", "web_id": 1, "url": "u", "title": "t", "text": "body"}])

def test_factory_builds_dense():
    cfg = RetrievalConfig(retriever_mode="dense")
    r = build_retriever(_chunks(), np.array([[1.0, 0.0]], dtype="float32"), FakeEmbedder(), cfg)
    assert isinstance(r, DenseRetriever)
    assert len(r.search("q", k=1)) == 1

def test_factory_rejects_unknown_mode():
    import pytest
    cfg = RetrievalConfig(retriever_mode="bogus")
    with pytest.raises(ValueError):
        build_retriever(_chunks(), np.array([[1.0, 0.0]], dtype="float32"), FakeEmbedder(), cfg)
