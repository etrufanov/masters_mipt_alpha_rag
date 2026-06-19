"""Dense: top-k фрагментов по косинусной близости эмбеддингов BGE-M3."""
import numpy as np
import pandas as pd
from rag.schema import Retrieved


class DenseRetriever:
    """Ищет фрагменты по косинусу между вектором запроса и матрицей эмбеддингов корпуса."""

    def __init__(self, chunks: pd.DataFrame, embeddings: np.ndarray, embedder):
        self.chunks = chunks.reset_index(drop=True)
        self.embeddings = embeddings
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[Retrieved]:
        """Вернуть top-k фрагментов по убыванию косинуса."""
        q = self.embedder.encode_queries([query])[0]
        scores = self.embeddings @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            row = self.chunks.iloc[int(i)]
            out.append(Retrieved(chunk_id=row.chunk_id, web_id=int(row.web_id),
                                 url=row.url, title=row.title, text=row.text,
                                 score=float(scores[i])))
        return out
