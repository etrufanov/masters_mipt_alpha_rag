"""Базовые структуры данных пайплайна: фрагмент корпуса и результат поиска."""
from dataclasses import dataclass

@dataclass
class Chunk:
    """Фрагмент документа базы знаний (единица индексации)."""
    chunk_id: str
    web_id: int
    url: str
    title: str
    text: str

@dataclass
class Retrieved(Chunk):
    """Фрагмент, найденный ретривером, вместе с его релевантностью.

    `score` — это либо косинус (dense), либо вес RRF, либо сигмоида реранкера
    в зависимости от того, какой ретривер вернул фрагмент.
    """
    score: float = 0.0
