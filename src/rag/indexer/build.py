"""Сборка поискового индекса: очистка → чанкинг → эмбеддинги → запись в artifacts/."""
import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from pathlib import Path
from rag.schema import Chunk
from rag.indexer.clean import compute_boilerplate, clean_text, drop_junk
from rag.indexer.chunk import chunk_text


def build_chunks(websites: pd.DataFrame, target=400, overlap=64) -> list[Chunk]:
    """Превратить таблицу страниц в список фрагментов: чистит шаблонный текст и режет на чанки."""
    raw = websites["text"].fillna("").tolist()
    boilerplate = compute_boilerplate(raw, min_doc_freq=5)
    out: list[Chunk] = []
    for _, row in websites.iterrows():
        cleaned = clean_text(row["text"] if isinstance(row["text"], str) else "", boilerplate)
        cleaned = drop_junk(cleaned, min_len=20)
        if cleaned is None:
            continue
        title = str(row.get("title") or "")
        for i, ck in enumerate(
            chunk_text(cleaned, target=target, overlap=overlap, len_fn=lambda s: len(s.split()))
        ):
            out.append(
                Chunk(
                    chunk_id=f"{row['web_id']}:{i}",
                    web_id=int(row["web_id"]),
                    url=str(row.get("url") or ""),
                    title=title,
                    text=ck,
                )
            )
    return out


def embed_text_for(chunk: Chunk) -> str:
    """Текст для эмбеддинга: заголовок + тело (заголовок добавляет контекст для поиска)."""
    return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text


def build_index(
    websites: pd.DataFrame,
    embedder,
    out_dir: Path,
    target=400,
    overlap=64,
) -> None:
    """Построить индекс и сохранить в out_dir: embeddings.npy, chunks.parquet, meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(websites, target=target, overlap=overlap)
    embs = embedder.encode_passages([embed_text_for(c) for c in chunks])
    np.save(out_dir / "embeddings.npy", embs)
    pd.DataFrame([asdict(c) for c in chunks]).to_parquet(out_dir / "chunks.parquet")
    (out_dir / "meta.json").write_text(
        json.dumps({"n_chunks": len(chunks), "dim": int(embs.shape[1])})
    )
