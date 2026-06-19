"""Фабрика ретривера: собирает dense или hybrid по RetrievalConfig (единая точка сборки)."""
import numpy as np, pandas as pd
from rag.config import RetrievalConfig
from rag.retriever.dense import DenseRetriever
from rag.retriever.bm25 import BM25Retriever
from rag.retriever.hybrid import HybridRetriever, CrossEncoderReranker

def build_retriever(chunks: pd.DataFrame, embeddings: np.ndarray, embedder, rcfg: RetrievalConfig):
    """Собрать ретривер по конфигу."""
    dense = DenseRetriever(chunks, embeddings, embedder)
    # Для MVP, потом ушли в hybrid
    if rcfg.retriever_mode == "dense":
        return dense
    if rcfg.retriever_mode == "hybrid":
        bm25 = BM25Retriever(chunks)
        reranker = CrossEncoderReranker(rcfg.rerank_model)
        return HybridRetriever(dense, bm25, reranker,
                               k_dense=rcfg.k_dense_fuse, k_bm25=rcfg.k_bm25, rrf_k=rcfg.rrf_k)
    raise ValueError(f"unknown retriever_mode: {rcfg.retriever_mode}")
