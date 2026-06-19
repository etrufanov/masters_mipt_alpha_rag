"""Дисковый кэш (json файл) найденных контекстов с ключом по q_id.

Для тюнинга промпта генерации
"""
import json
from pathlib import Path
from rag.schema import Retrieved

# Только поля, влияющие на поиск; конфиг генерации (промпт, max_tokens, порог) на кэш не влияет.
_SIG_FIELDS = ("retriever_mode", "embed_model", "final_n", "k_dense",
               "k_bm25", "k_dense_fuse", "rrf_k", "rerank_model")


def config_signature(rcfg) -> str:
    """Ключ инвалидации кэша."""
    return "|".join(f"{f}={getattr(rcfg, f)}" for f in _SIG_FIELDS)


def cached_search(retriever_factory, q_ids, queries, k, cache_path, sig, progress=True):
    """Вернуть top-k контекстов на запрос, переиспользуя JSON-кэш с ключом q_id, валидный для `sig`.

    retriever_factory — функция без аргументов, собирающая ретривер; вызывается ТОЛЬКО
    когда часть q_id отсутствует, поэтому при полном попадании в кэш эмбеддер/реранкер не грузятся.
    """
    cache_path = Path(cache_path)
    items: dict[str, list[dict]] = {}
    if cache_path.exists():
        blob = json.loads(cache_path.read_text())
        if blob.get("sig") == sig:
            items = blob.get("items", {})

    missing = [(qid, q) for qid, q in zip(q_ids, queries) if str(qid) not in items]
    if missing:
        retriever = retriever_factory()
        it = missing
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(missing, desc="retrieve", unit="q")
            except ImportError:
                pass
        for qid, q in it:
            items[str(qid)] = [c.__dict__ for c in retriever.search(q, k=k)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"sig": sig, "items": items}, ensure_ascii=False))

    return [[Retrieved(**d) for d in items[str(qid)]] for qid in q_ids]
