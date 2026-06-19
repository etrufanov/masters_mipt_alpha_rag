"""Метрики качества решения об abstain (precision/recall/F1) против эталонных меток."""
from rag.generator.abstain import is_abstain

def abstain_scores(preds: list[str], refs: list[str]) -> dict:
    """Посчитать precision/recall/F1 abstain, считая положительным классом отказ `Нет ответа.`"""
    pred_a = [is_abstain(p) for p in preds]
    ref_a = [is_abstain(r) for r in refs]
    tp = sum(p and r for p, r in zip(pred_a, ref_a))
    fp = sum(p and not r for p, r in zip(pred_a, ref_a))
    fn = sum((not p) and r for p, r in zip(pred_a, ref_a))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
