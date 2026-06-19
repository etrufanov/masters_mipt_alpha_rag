import pandas as pd
from rag.eval.abstain_eval import abstain_scores
from rag.eval.devset import build_devset

def test_abstain_scores_perfect():
    preds = ["Нет ответа.", "счёт 20 знаков"]
    refs  = ["Нет ответа.", "длинный реальный ответ про счёт"]
    s = abstain_scores(preds, refs)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["f1"] == 1.0

def test_abstain_scores_counts_mistakes():
    preds = ["Нет ответа.", "Нет ответа."]   # 2nd is wrong abstain
    refs  = ["Нет ответа.", "реальный ответ"]
    s = abstain_scores(preds, refs)
    assert s["precision"] == 0.5   # 1 of 2 predicted-abstain correct

def test_build_devset_stratified():
    q = pd.DataFrame({"q_id": range(10), "query": [f"q{i}" for i in range(10)]})
    s = pd.DataFrame({"q_id": range(10),
                      "answer_new": ["Нет ответа."]*4 + ["короткий"]*2 + ["длинный "*20]*4})
    dev = build_devset(q, s, n=6, seed=0)
    assert len(dev) == 6
    assert {"q_id", "query", "answer_new"}.issubset(dev.columns)
