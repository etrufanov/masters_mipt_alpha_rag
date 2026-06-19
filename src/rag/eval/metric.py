"""Метрика соревнования: Score = BERTScore-recall * L (штраф за длину)."""
from rag.generator.length import length_unit_count

def length_gate(r: float) -> float:
    """Множитель длины L по отношению r = len(ответ)/len(эталон): 1 при r<=1.5, линейно до 0 при r=3."""
    if r <= 1.5:
        return 1.0
    if r < 3.0:
        return -2.0 / 3.0 * r + 2.0
    return 0.0

def length_ratio(ours: str, ref: str, unit: str) -> float:
    """Отношение длины нашего ответа к эталону (знаменатель не меньше 1, чтобы не делить на ноль)."""
    denom = max(length_unit_count(ref, unit), 1)
    return length_unit_count(ours, unit) / denom

def combined_score(recall: float, ours: str, ref: str, unit: str = "words") -> float:
    """Итоговый скор одной пары: recall, умноженный на L-gate по длине."""
    return recall * length_gate(length_ratio(ours, ref, unit))

def bertscore_recall(
    cands: list[str], refs: list[str], lang: str = "ru", device: str = "cpu"
) -> list[float]:
    """Реальная метрика — скачивает BERT-модель. Используется только в evaluate.py / smoke-тестах.

    По умолчанию считается на CPU, чтобы не конкурировать за VRAM с vLLM в одном процессе
    (vLLM держит бóльшую часть видеопамяти; загрузка BERT на GPU вызвала бы OOM)."""
    from bert_score import score as bs
    _, R, _ = bs(cands, refs, lang=lang, rescale_with_baseline=False, device=device)
    return R.tolist()
