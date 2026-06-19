"""Smoke-прогон на 12 стратифицированных вопросах: глазами проверить качество и abstain.

Печатает вопрос, скор top-1, флаги abstain и обрезанные ответ/эталон для ручного контроля.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rag.config import Paths, RetrievalConfig, GenerationConfig
from rag.indexer.embed import Embedder
from rag.retriever.factory import build_retriever
from rag.generator.llm import VLLMModel
from rag.generator.generator import Generator
from rag.eval.devset import build_devset
from rag.generator.abstain import is_abstain


def main():
    """Прогнать 12 вопросов через ретривер+генератор и распечатать результаты для визуальной проверки."""
    paths, rcfg, gcfg = Paths(), RetrievalConfig(), GenerationConfig()
    q = pd.read_csv(paths.questions)
    sample = pd.read_csv(paths.sample)
    dev = build_devset(q, sample, n=12)

    chunks = pd.read_parquet(paths.artifacts / "chunks.parquet")
    embs = np.load(paths.artifacts / "embeddings.npy")
    retr = build_retriever(chunks, embs, Embedder(rcfg.embed_model), rcfg)
    gen = Generator(VLLMModel(gcfg.llm_model), gcfg)

    for _, row in dev.iterrows():
        ctx = retr.search(row["query"], rcfg.final_n)
        top1 = ctx[0].score if ctx else 0.0
        pred = gen.answer(row["query"], ctx)
        ref = str(row["answer_new"])
        print("=" * 90)
        print("Q   :", row["query"])
        print("top1: %.3f   pred_abstain=%s  ref_abstain=%s" % (top1, is_abstain(pred), is_abstain(ref)))
        print("PRED:", pred[:320])
        print("REF :", ref[:320])


if __name__ == "__main__":
    main()
