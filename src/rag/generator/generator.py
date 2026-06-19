"""Генератор ответов: строит промпт, зовёт LLM и нормализует вывод (включая abstain)."""
from rag.config import GenerationConfig
from rag.generator.abstain import ABSTAIN, normalize_output
from rag.generator.prompt import build_prompt
from rag.schema import Retrieved

class Generator:
    """Превращает (запрос, контексты) в ответ; решение об abstain в основном за LLM."""

    def __init__(self, llm, cfg: GenerationConfig):
        self.llm = llm
        self.cfg = cfg

    def _retrieval_abstain(self, ctx: list[Retrieved]) -> bool:
        """Жёсткий abstain по скору поиска ещё до LLM. При threshold=0.0 срабатывает только на пустом ctx."""
        return not ctx or ctx[0].score < self.cfg.abstain_threshold

    def answer(self, query: str, ctx: list[Retrieved]) -> str:
        """Ответить на один запрос (обёртка над answer_batch)."""
        return self.answer_batch([query], [ctx])[0]

    def answer_batch(self, queries: list[str], ctxs: list[list[Retrieved]]) -> list[str]:
        """Ответить пачкой: запросы с пустым контекстом сразу abstain, остальные — одним батчем в LLM."""
        results: list[str | None] = [None] * len(queries)
        to_gen: list[int] = []
        prompts: list[str] = []
        for i, (q, ctx) in enumerate(zip(queries, ctxs)):
            if self._retrieval_abstain(ctx):
                results[i] = ABSTAIN
            else:
                to_gen.append(i)
                prompts.append(build_prompt(q, ctx))
        if prompts:
            gen = self.llm.generate(prompts, max_tokens=self.cfg.max_tokens,
                                    temperature=self.cfg.temperature, stop=None)
            for idx, text in zip(to_gen, gen):
                results[idx] = normalize_output(text)
        return [r if r is not None else ABSTAIN for r in results]
