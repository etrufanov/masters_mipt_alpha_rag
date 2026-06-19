"""Обёртка над эмбеддером BGE-M3 для построения dense векторов фрагментов и запросов."""
import numpy as np


class Embedder:
    """Обёртка над sentence-transformers BGE-M3 (dense векторы, L2-нормированные)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Закодировать тексты в L2-нормированные float32-векторы (косинус = скалярное произведение)."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=show_progress,
        ).astype("float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Закодировать фрагменты корпуса."""
        return self._encode(texts, show_progress=True)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Закодировать запросы."""
        return self._encode(texts, show_progress=False)
