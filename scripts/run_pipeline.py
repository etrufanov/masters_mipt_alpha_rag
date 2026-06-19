"""Полный прогон: поиск (с кэшем по q_id) + генерация по всем 6977 вопросам -> submission.csv.

Запускать на GPU после build_index. 2 часа на H100 (генерация — основная стоимость;
поиск переиспользуется из кэша при повторных запусках).
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rag.config import Paths, RetrievalConfig, GenerationConfig
from rag.indexer.embed import Embedder
from rag.retriever.factory import build_retriever
from rag.retriever.cache import cached_search, config_signature
from rag.generator.llm import VLLMModel
from rag.generator.generator import Generator
from rag.pipeline import Pipeline

def main():
    """Найти контексты для всех вопросов, сгенерировать ответы и записать submission.csv."""
    paths, rcfg, gcfg = Paths(), RetrievalConfig(), GenerationConfig()
    chunks = pd.read_parquet(paths.artifacts / "chunks.parquet")
    embs = np.load(paths.artifacts / "embeddings.npy")
    questions = pd.read_csv(paths.questions)

    ctxs = cached_search(
        lambda: build_retriever(chunks, embs, Embedder(rcfg.embed_model), rcfg),
        questions["q_id"].tolist(), questions["query"].fillna("").tolist(),
        k=rcfg.final_n, cache_path=paths.artifacts / "retrieval_cache.json",
        sig=config_signature(rcfg))

    gen = Generator(VLLMModel(gcfg.llm_model, gpu_memory_utilization=gcfg.gpu_mem_util), gcfg)
    pipe = Pipeline(None, gen, final_n=rcfg.final_n)
    sub = pipe.run(questions, ctxs=ctxs)
    sub.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv ({len(sub)} rows)")

if __name__ == "__main__":
    main()
