import numpy as np, pandas as pd
from rag.indexer.build import build_chunks

class FakeEmbedder:
    dim = 4
    def encode_passages(self, texts):
        return np.ones((len(texts), self.dim), dtype="float32")

def test_build_chunks_produces_rows_with_ids_and_metadata():
    df = pd.DataFrame([
        {"web_id": 1, "url": "u1", "kind": "html", "title": "Счета",
         "text": "Номер счёта состоит из двадцати знаков. " * 30},
        {"web_id": 2, "url": "u2", "kind": "html", "title": "Junk", "text": "x"},  # dropped
    ])
    chunks = build_chunks(df, target=50, overlap=10)
    assert len(chunks) >= 1
    assert all(c.chunk_id.startswith("1:") for c in chunks)   # doc 2 dropped as junk
    assert chunks[0].title == "Счета"
