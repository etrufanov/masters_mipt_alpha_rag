"""Конфигурация пайплайна: пути к данным, настройки ретривера и генератора."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Paths:
    inputs: Path = ROOT / "inputs" / "inputs"
    artifacts: Path = ROOT / "artifacts"

    @property
    def questions(self) -> Path:
        return self.inputs / "questions.csv"

    @property
    def websites(self) -> Path:
        return self.inputs / "websites.csv"

    @property
    def sample(self) -> Path:
        return self.inputs / "sample_submission.csv"


@dataclass
class RetrievalConfig:
    embed_model: str = "BAAI/bge-m3"
    k_dense: int = 20
    final_n: int = 5
    # "dense" | "hybrid"
    retriever_mode: str = "hybrid"
    k_bm25: int = 30
    k_dense_fuse: int = 30
    rrf_k: int = 60
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class GenerationConfig:
    llm_model: str = "t-tech/T-pro-it-2.1"
    max_tokens: int = 128
    temperature: float = 0.0
    abstain_threshold: float = 0.0
    len_unit: str = "words"
    # Оставляем для reranker (5.5 GB)
    gpu_mem_util: float = 0.80
