"""Сборка dev-подвыборки для офлайн-оценки: стратификация по типу эталонного ответа."""
import pandas as pd
from rag.generator.abstain import is_abstain

def _bucket(ans: str) -> str:
    """Тип эталона для стратификации: abstain / short (<=35 слов) / long."""
    if is_abstain(ans):
        return "abstain"
    return "short" if len((ans or "").split()) <= 35 else "long"

def build_devset(questions: pd.DataFrame, sample: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Собрать n вопросов, стратифицированных по типу эталона (abstain/short/long), детерминированно по seed."""
    df = questions.merge(sample, on="q_id")
    df["_bucket"] = df["answer_new"].fillna("").map(_bucket)
    per = max(1, n // df["_bucket"].nunique())
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("_bucket")]
    out = pd.concat(parts).sample(frac=1, random_state=seed).head(n)
    return out.drop(columns="_bucket").reset_index(drop=True)
