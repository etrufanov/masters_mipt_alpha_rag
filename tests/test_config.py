from rag.config import RetrievalConfig, GenerationConfig, Paths

def test_defaults():
    assert RetrievalConfig().k_dense == 20
    assert RetrievalConfig().final_n == 5
    assert GenerationConfig().max_tokens == 128
    assert 0.0 <= GenerationConfig().abstain_threshold <= 1.0
    assert Paths().questions.name == "questions.csv"
