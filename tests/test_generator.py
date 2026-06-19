from rag.generator.generator import Generator
from rag.generator.llm import FakeLLM
from rag.generator.abstain import ABSTAIN
from rag.config import GenerationConfig
from rag.schema import Retrieved

def _ctx(score):
    return [Retrieved(chunk_id="1:0", web_id=1, url="u", title="t", text="body", score=score)]

def test_abstains_when_no_context():
    g = Generator(FakeLLM(["irrelevant"]), GenerationConfig())
    assert g.answer("q", []) == ABSTAIN

def test_abstains_when_top_score_below_threshold():
    cfg = GenerationConfig(abstain_threshold=0.5)
    g = Generator(FakeLLM(["реальный ответ"]), cfg)
    assert g.answer("q", _ctx(0.4)) == ABSTAIN

def test_answers_when_confident():
    cfg = GenerationConfig(abstain_threshold=0.5)
    g = Generator(FakeLLM(["реальный ответ"]), cfg)
    assert g.answer("q", _ctx(0.9)) == "реальный ответ"

def test_llm_abstain_phrase_normalized():
    cfg = GenerationConfig(abstain_threshold=0.5)
    g = Generator(FakeLLM(["нет ответа"]), cfg)
    assert g.answer("q", _ctx(0.9)) == ABSTAIN

def test_batch_reassembles_in_order():
    cfg = GenerationConfig(abstain_threshold=0.5)
    # second item is below threshold -> abstain without calling llm for it
    g = Generator(FakeLLM(["A", "C"]), cfg)
    out = g.answer_batch(["q1", "q2", "q3"], [_ctx(0.9), _ctx(0.1), _ctx(0.9)])
    assert out == ["A", ABSTAIN, "C"]
