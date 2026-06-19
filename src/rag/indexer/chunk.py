"""Разбиение текста на фрагменты по сегментам (абзацы/предложения) с перекрытием."""
import re

_SENT = re.compile(r"(?<=[.!?])\s+")   # граница предложений: после .!? и пробел

def _segments(text: str) -> list[str]:
    """Разбить текст на атомарные сегменты: абзацы, а длинные/многострочные — на предложения."""
    segs: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if "\n" in para or len(para) > 400:
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                segs.extend(s for s in _SENT.split(line) if s)
        else:
            segs.append(para)
    return segs

def chunk_text(text, target=400, overlap=64, len_fn=len) -> list[str]:
    """Склеить сегменты в чанки до`target` единиц длины, с хвостом `overlap` между соседними.

    `len_fn` задаёт единицу длины (по умолчанию символы; индекс строится по словам).
    Перекрытие сохраняет контекст на стыке чанков, чтобы ответ не разрезало границей.
    """
    text = (text or "").strip()
    if not text:
        return []
    segs = _segments(text)
    chunks, cur, cur_len = [], [], 0
    for seg in segs:
        slen = len_fn(seg)
        if cur and cur_len + slen > target:
            chunks.append(" ".join(cur))
            tail, tlen = [], 0
            for s in reversed(cur):
                if tlen >= overlap:
                    break
                tail.insert(0, s); tlen += len_fn(s)
            cur, cur_len = tail[:], sum(len_fn(s) for s in tail)
        cur.append(seg); cur_len += slen
    if cur:
        chunks.append(" ".join(cur))
    return chunks
