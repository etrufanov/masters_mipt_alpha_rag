"""Очистка текста страниц: удаление шаблонных строк (меню/футеры), нормализация пробелов."""
import re
from collections import Counter

_WS_LINE = re.compile(r"[ \t]+")    # схлопывание подряд идущих пробелов/табов
_MULTI_NL = re.compile(r"\n{3,}")   # три+ пустых строки -> один разделитель абзацев

def compute_boilerplate(docs, min_doc_freq: int) -> set[str]:
    """Найти шаблонные строки — короткие строки, встречающиеся в >= min_doc_freq документах.

    Это навигация/футеры, повторяющиеся по всему сайту; их вырезаем, чтобы не засорять индекс.
    """
    counts = Counter()
    for d in docs:
        for line in {l.strip() for l in (d or "").splitlines() if l.strip()}:
            counts[line] += 1
    return {line for line, n in counts.items() if n >= min_doc_freq and len(line) < 60}

def clean_text(text: str, boilerplate: set[str]) -> str:
    """Убрать шаблонные строки и нормализовать пробелы/переводы строк в одном документе."""
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped in boilerplate:
            continue
        lines.append(_WS_LINE.sub(" ", line).rstrip())
    out = "\n".join(lines)
    return _MULTI_NL.sub("\n\n", out).strip()

def drop_junk(text: str, min_len: int = 20) -> str | None:
    """Отбросить слишком короткие документы (< min_len символов) — вернуть None для пропуска."""
    t = (text or "").strip()
    return t if len(t) >= min_len else None
