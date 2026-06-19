"""Протокол ретривера — общий интерфейс для dense / bm25 / hybrid."""
from typing import Protocol
from rag.schema import Retrieved


class Retriever(Protocol):
    """Любой ретривер умеет одно: по запросу вернуть top-k найденных фрагментов."""

    def search(self, query: str, k: int) -> list[Retrieved]: ...
