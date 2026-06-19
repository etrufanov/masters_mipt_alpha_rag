"""Каноническая строка отказа `Нет ответа.` и распознавание/нормализация abstain-ответов."""

ABSTAIN = "Нет ответа."

def _norm(text: str) -> str:
    """Нормализовать для сравнения: обрезать пробелы и точки, привести к нижнему регистру."""
    return (text or "").strip().rstrip(".").strip().lower()

def is_abstain(text: str) -> bool:
    """True, если текст — ровно канонический отказ `Нет ответа.` (с точностью до регистра/точек)."""
    return _norm(text) == "нет ответа"

def normalize_output(text: str) -> str:
    """Привести вывод модели к канону: пустой или отказ -> ABSTAIN, иначе текст без лишних пробелов."""
    t = (text or "").strip()
    if not t or is_abstain(t):
        return ABSTAIN
    return t
