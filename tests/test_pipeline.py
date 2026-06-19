import numpy as np, pandas as pd
from rag.pipeline import Pipeline
from rag.retriever.dense import DenseRetriever
from rag.generator.generator import Generator
from rag.generator.llm import FakeLLM
from rag.generator.abstain import ABSTAIN
from rag.config import GenerationConfig

class FakeEmbedder:
    def encode_queries(self, texts):
        return np.array([[1.0, 0.0]], dtype="float32")

def test_pipeline_produces_submission_columns():
    chunks = pd.DataFrame([{"chunk_id": "1:0", "web_id": 1, "url": "u", "title": "t", "text": "body"}])
    embs = np.array([[1.0, 0.0]], dtype="float32")
    retr = DenseRetriever(chunks, embs, FakeEmbedder())
    gen = Generator(FakeLLM(["реальный ответ", "реальный ответ"]), GenerationConfig(abstain_threshold=0.5))
    pipe = Pipeline(retr, gen, final_n=5)
    questions = pd.DataFrame({"q_id": [1, 2], "query": ["вопрос1", "вопрос2"]})
    sub = pipe.run(questions)
    assert list(sub.columns) == ["q_id", "answer_new"]
    assert len(sub) == 2
    assert sub.iloc[0].answer_new in ("реальный ответ", ABSTAIN)
