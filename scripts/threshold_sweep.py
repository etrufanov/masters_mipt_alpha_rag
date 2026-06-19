"""Настройка порога abstain по сохранённым dev-предсказаниям. Без vLLM.

Не зависит от режима ретривера: использует сохранённый по строкам top-1 поиска (косинус dense
ИЛИ сигмоида реранкера) — ровно ту величину, с которой abstain-гейт сравнивается в generator.py.
Требует, чтобы dev-прогон шёл с abstain_threshold=0.0 (гейт открыт) и все строки были сгенерированы;
повышение T здесь может только ДОБАВИТЬ отказов, пере-оценивая перевёрнутые строки как ABSTAIN.
Выбирать T так, чтобы abst_rate совпадал с ref_abstain (32%) и false_ans/false_abst были сбалансированы —
НЕ гнаться за `mean` (он переоценивает).
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rag.config import Paths, GenerationConfig
from rag.eval.metric import bertscore_recall, length_ratio, length_gate
from rag.generator.abstain import ABSTAIN, is_abstain


def score_of(preds, refs, recalls, unit):
    """Векторный скор по парам: recall * L-gate для каждой пары (наш ответ, эталон)."""
    return np.array([r * length_gate(length_ratio(p, ref, unit))
                     for p, ref, r in zip(preds, refs, recalls)])


def main():
    """Перебрать пороги T по dev_preds.parquet и для каждого распечатать abst_rate / false_ans / false_abst."""
    paths, gcfg = Paths(), GenerationConfig()
    unit = gcfg.len_unit
    d = pd.read_parquet(paths.artifacts / "dev_preds.parquet")
    refs = d["answer_new"].fillna("").tolist()
    preds = d["pred"].tolist()
    top1 = d["retr_top1"].to_numpy()

    rec_cur = np.array(bertscore_recall(preds, refs))
    rec_ab = np.array(bertscore_recall([ABSTAIN] * len(refs), refs))
    s_cur = score_of(preds, refs, rec_cur, unit)
    s_ab = score_of([ABSTAIN] * len(refs), refs, rec_ab, unit)

    ref_ab = np.array([is_abstain(r) for r in refs])
    pred_ab = np.array([is_abstain(p) for p in preds])
    n = len(refs)

    print(f"rows={n}  ref_abstain={int(ref_ab.sum())} ({ref_ab.mean():.0%})  "
          f"model_self_abstain={int(pred_ab.sum())}  top1[min/med/max]="
          f"{top1.min():.2f}/{np.median(top1):.2f}/{top1.max():.2f}")
    print("  T     mean    abst_rate  false_ans  false_abst  forced+")
    for T in np.round(np.arange(0.05, 0.95, 0.05), 2):
        force = top1 < T
        new_ab = pred_ab | force
        s = np.where(force & ~pred_ab, s_ab, s_cur)
        abst_rate = new_ab.mean()
        false_ans = int((~new_ab & ref_ab).sum())
        false_abst = int((new_ab & ~ref_ab).sum())
        added = int((force & ~pred_ab).sum())
        print(f" {T:.2f}   {s.mean():.4f}    {abst_rate:5.1%}      {false_ans:3d}        "
              f"{false_abst:3d}        {added:3d}")


if __name__ == "__main__":
    main()
