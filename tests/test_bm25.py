import pandas as pd
from rag.retriever.bm25 import BM25Retriever

def _store():
    return pd.DataFrame([
        {"chunk_id": "1:0", "web_id": 1, "url": "u", "title": "t", "text": "номер счёта двадцать знаков"},
        {"chunk_id": "2:0", "web_id": 2, "url": "u", "title": "t", "text": "кредитная карта оформление"},
    ])

def test_bm25_ranks_lexical_match_first():
    r = BM25Retriever(_store())
    out = r.search("счёт номер", k=2)
    assert out[0].chunk_id == "1:0"
    assert out[0].score >= out[1].score
