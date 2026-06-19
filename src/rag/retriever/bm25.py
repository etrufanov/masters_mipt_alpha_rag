"""Лексический поиск BM25Okapi: top-k фрагментов по точному совпадению слов."""
import re
import pandas as pd
from rank_bm25 import BM25Okapi
from rag.schema import Retrieved

_TOK = re.compile(r"\w+", re.UNICODE)

def _tok(s: str) -> list[str]:
    """Простая токенизация: lower."""
    return _TOK.findall((s or "").lower())

class BM25Retriever:
    """Ретривер BM25: ловит точные совпадения терминов, которые dense может упустить."""

    def __init__(self, chunks: pd.DataFrame):
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = BM25Okapi([_tok(t) for t in self.chunks["text"].tolist()])

    def search(self, query: str, k: int) -> list[Retrieved]:
        """Вернуть top-k фрагментов по убыванию BM25-скора запроса."""
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in order:
            row = self.chunks.iloc[i]
            out.append(Retrieved(chunk_id=row.chunk_id, web_id=int(row.web_id),
                                 url=row.url, title=row.title, text=row.text,
                                 score=float(scores[i])))
        return out
