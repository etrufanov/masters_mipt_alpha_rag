"""Офлайн-оценка на dev (n=300): abstain P/R/F1, средний скор и 4-квадрантная разбивка.

ВАЖНО: смотреть на abstain rate и разбивку, а не на средний скор.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rag.config import Paths, RetrievalConfig, GenerationConfig
from rag.indexer.embed import Embedder
from rag.retriever.factory import build_retriever
from rag.retriever.cache import cached_search, config_signature
from rag.generator.llm import VLLMModel
from rag.generator.generator import Generator
from rag.eval.devset import build_devset
from rag.eval.abstain_eval import abstain_scores
from rag.eval.metric import bertscore_recall, combined_score, length_ratio, length_gate
from rag.generator.abstain import ABSTAIN, is_abstain


def main(n=300):
    """Прогнать пайплайн на dev-подвыборке и напечатать abstain-метрики + разбивку по квадрантам."""
    paths, rcfg, gcfg = Paths(), RetrievalConfig(), GenerationConfig()
    questions = pd.read_csv(paths.questions)
    sample = pd.read_csv(paths.sample)
    dev = build_devset(questions, sample, n=n)
    chunks = pd.read_parquet(paths.artifacts / "chunks.parquet")
    embs = np.load(paths.artifacts / "embeddings.npy")
    queries = dev["query"].fillna("").tolist()
    ctxs = cached_search(
        lambda: build_retriever(chunks, embs, Embedder(rcfg.embed_model), rcfg),
        dev["q_id"].tolist(), queries, k=rcfg.final_n,
        cache_path=paths.artifacts / "retrieval_cache.json", sig=config_signature(rcfg))
    top1 = [c[0].score if c else 0.0 for c in ctxs]   # скор top-1, который видит abstain-гейт

    gen = Generator(VLLMModel(gcfg.llm_model, gpu_memory_utilization=gcfg.gpu_mem_util), gcfg)
    preds = gen.answer_batch(queries, ctxs)
    refs = dev["answer_new"].fillna("").tolist()

    dump = dev[["q_id", "query", "answer_new"]].copy()
    dump["pred"] = preds
    dump["retr_top1"] = top1
    dump.to_parquet(paths.artifacts / "dev_preds.parquet")

    print("ABSTAIN:", abstain_scores(preds, refs))

    unit = gcfg.len_unit
    recalls = bertscore_recall(preds, refs)
    rows = []
    for p, ref, rec in zip(preds, refs, recalls):
        ratio = length_ratio(p, ref, unit)
        L = length_gate(ratio)
        rows.append((rec, ratio, L, rec * L, is_abstain(p), is_abstain(ref)))
    df = pd.DataFrame(rows, columns=["recall", "ratio", "L", "score", "pred_ab", "ref_ab"])
    print(f"mean simulated score: {df.score.mean():.4f}")

    aa = bertscore_recall([ABSTAIN] * len(refs), refs)
    aa_score = np.mean([r * length_gate(length_ratio(ABSTAIN, ref, unit))
                        for r, ref in zip(aa, refs)])
    print(f"(all-abstain baseline: {aa_score:.4f})")

    def show(name, mask):
        """Напечатать средние recall/L/ratio/score по одному квадранту (подмножеству строк)."""
        s = df[mask]
        if len(s):
            print(f"  {name:38s} n={len(s):3d}  recall={s.recall.mean():.3f}  "
                  f"L={s.L.mean():.3f}  ratio_p50={s.ratio.median():.2f}  score={s.score.mean():.3f}")

    print("breakdown:")
    show("answered, ref answerable", (~df.pred_ab) & (~df.ref_ab))
    show("answered, ref ABSTAIN (false answer)", (~df.pred_ab) & (df.ref_ab))
    show("abstained, ref answerable (false ab)", (df.pred_ab) & (~df.ref_ab))
    show("abstained, ref ABSTAIN (correct)", (df.pred_ab) & (df.ref_ab))

    ans = df[~df.pred_ab]
    if len(ans):
        print(f"answered length ratio: p50={ans.ratio.median():.2f} "
              f"p75={ans.ratio.quantile(.75):.2f} p90={ans.ratio.quantile(.9):.2f}  "
              f"frac ratio>1.5: {(ans.ratio > 1.5).mean():.2f}")


if __name__ == "__main__":
    main()
