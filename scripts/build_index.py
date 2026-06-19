"""Скрипт сборки индекса: читает websites.csv и пишет artifacts/ (эмбеддинги, чанки, мета).

Запускать на GPU.
"""
import sys; sys.path.insert(0, "src")
import pandas as pd
from rag.config import Paths, RetrievalConfig
from rag.indexer.embed import Embedder
from rag.indexer.build import build_index

def main():
    """Загрузить страницы, построить эмбеддинги и сохранить индекс в artifacts/."""
    paths, rcfg = Paths(), RetrievalConfig()
    websites = pd.read_csv(paths.websites)
    embedder = Embedder(rcfg.embed_model)
    build_index(websites, embedder, paths.artifacts)
    print(f"index built at {paths.artifacts}")

if __name__ == "__main__":
    main()
