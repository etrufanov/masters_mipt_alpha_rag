"""Гибридный поиск: слияние dense+BM25 по RRF и переранжирование кросс-энкодером.

Схема: dense.search и bm25.search дают два ранжированных списка → RRF их объединяет
по позициям (не по сырым скорам, которые несравнимы) → реранкер bge-reranker-v2-m3
переоценивает объединённых кандидатов и оставляет top-k.
"""
import numpy as np
from collections import defaultdict
from rag.schema import Retrieved

def rrf_fuse(ranked_lists: list[list[Retrieved]], k: int = 60) -> list[Retrieved]:
    """Слить несколько ранжированных списков по Reciprocal Rank Fusion: вес = sum 1/(k + позиция).

    Работает с позициями, а не сырыми скорами, поэтому корректно объединяет косинус и BM25,
    которые лежат в разных шкалах.
    """
    scores: dict[str, float] = defaultdict(float)
    obj: dict[str, Retrieved] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item.chunk_id] += 1.0 / (k + rank + 1)
            obj.setdefault(item.chunk_id, item)
    fused = []
    for cid, sc in sorted(scores.items(), key=lambda x: -x[1]):
        it = obj[cid]
        fused.append(Retrieved(chunk_id=it.chunk_id, web_id=it.web_id, url=it.url,
                               title=it.title, text=it.text, score=sc))
    return fused

class CrossEncoderReranker:
    """Кросс-энкодер: переоценивает пары (запрос, фрагмент) и возвращает сигмоиду релевантности."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 512, batch_size: int = 64):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def rerank(self, query: str, cands: list[Retrieved], top_n: int) -> list[Retrieved]:
        """Переоценить кандидатов кросс-энкодером и вернуть top_n с сигмоидным скором в (0,1)."""
        pairs = [[query, c.text] for c in cands]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        probs = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype="float64")))
        ranked = sorted(zip(cands, probs), key=lambda x: -x[1])[:top_n]
        return [Retrieved(chunk_id=c.chunk_id, web_id=c.web_id, url=c.url, title=c.title,
                          text=c.text, score=float(s)) for c, s in ranked]

class HybridRetriever:
    """Связка dense + BM25 → RRF → реранкер: компромисс семантики и точного совпадения слов."""

    def __init__(self, dense, bm25, reranker, k_dense=50, k_bm25=50, rrf_k=60):
        self.dense, self.bm25, self.reranker = dense, bm25, reranker
        self.k_dense, self.k_bm25, self.rrf_k = k_dense, k_bm25, rrf_k

    def search(self, query: str, k: int) -> list[Retrieved]:
        """Найти top-k: взять кандидатов из dense и BM25, слить по RRF, переранжировать кросс-энкодером."""
        d = self.dense.search(query, self.k_dense)
        b = self.bm25.search(query, self.k_bm25)
        fused = rrf_fuse([d, b], k=self.rrf_k)
        return self.reranker.rerank(query, fused, top_n=k)
