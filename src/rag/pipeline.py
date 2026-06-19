"""Связка ретривер → генератор: на вход вопросы, на выход таблица ответов для сабмита."""
import pandas as pd

class Pipeline:
    """Прогоняет вопросы через поиск и генерацию, возвращает DataFrame `q_id, answer_new`."""

    def __init__(self, retriever, generator, final_n: int = 5):
        # retriever может быть None, если контексты уже посчитаны заранее (см. кэш в run_pipeline)
        self.retriever = retriever
        self.generator = generator
        self.final_n = final_n

    def run(self, questions: pd.DataFrame, ctxs=None) -> pd.DataFrame:
        """Сгенерировать ответы. `ctxs` — заранее найденные контексты; если None — ищем сами."""
        queries = questions["query"].fillna("").tolist()
        if ctxs is None:
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(it, **kw):
                    return it
            ctxs = [self.retriever.search(q, k=self.final_n)
                    for q in tqdm(queries, desc="retrieve", unit="q")]
        answers = self.generator.answer_batch(queries, ctxs)
        return pd.DataFrame({"q_id": questions["q_id"].tolist(), "answer_new": answers})
