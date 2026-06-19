"""Подсчёт длины ответа в выбранных единицах (словах или символах) для L-gate метрики."""

def length_unit_count(text: str, unit: str) -> int:
    """Длина текста: число слов при unit=='words', иначе число символов."""
    return len((text or "").split()) if unit == "words" else len(text or "")
