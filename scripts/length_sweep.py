"""Настройка длины: обрезает сохранённые dev-предсказания до разных лимитов слов и пересчитывает скор.

Приближает эффект краткого промпта без перезапуска vLLM. Отказы не обрезаются.
cap=99999 == без обрезки.
"""
import sys; sys.path.insert(0, "src")
import pandas as pd
from rag.config import Paths, GenerationConfig
from rag.eval.metric import bertscore_recall, length_ratio, length_gate
from rag.generator.abstain import is_abstain


def truncate(p: str, cap: int) -> str:
    """Обрезать ответ до `cap` слов; отказы возвращаем как есть."""
    if is_abstain(p):
        return p
    w = p.split()
    return " ".join(w[:cap]) if len(w) > cap else p


def main():
    """Перебрать лимиты слов по сохранённым dev_preds.parquet и распечатать скор/recall/L для каждого."""
    paths, gcfg = Paths(), GenerationConfig()
    unit = gcfg.len_unit
    d = pd.read_parquet(paths.artifacts / "dev_preds.parquet")
    refs = d["answer_new"].fillna("").tolist()
    preds = d["pred"].tolist()

    print("cap    mean_score   ans_recall  ans_L   frac_ratio>1.5")
    for cap in [99999, 100, 70, 50, 40, 30, 25, 20, 15]:
        tp = [truncate(p, cap) for p in preds]
        rec = bertscore_recall(tp, refs)
        rows = []
        for p, ref, r in zip(tp, refs, rec):
            ratio = length_ratio(p, ref, unit)
            L = length_gate(ratio)
            rows.append((r * L, r, L, ratio, is_abstain(p)))
        df = pd.DataFrame(rows, columns=["score", "recall", "L", "ratio", "ab"])
        ans = df[~df.ab]
        print(f"{cap:5d}   {df.score.mean():.4f}      {ans.recall.mean():.3f}      "
              f"{ans.L.mean():.3f}   {(ans.ratio > 1.5).mean():.2f}")


if __name__ == "__main__":
    main()
