from rag.generator.llm import FakeLLM


def test_fakellm_returns_scripted_outputs():
    llm = FakeLLM(["ответ один", "ответ два"])
    out = llm.generate(["p1", "p2"], max_tokens=50, temperature=0.0, stop=None)
    assert out == ["ответ один", "ответ два"]


def test_fakellm_default_echo():
    llm = FakeLLM()
    assert llm.generate(["hello"], max_tokens=5, temperature=0.0, stop=None) == ["ECHO"]


def test_fakellm_truncates_to_prompt_count():
    llm = FakeLLM(["a", "b", "c"])
    assert llm.generate(["p1"], max_tokens=5, temperature=0.0, stop=None) == ["a"]


def test_fakellm_asserts_when_too_few_outputs():
    import pytest
    llm = FakeLLM(["only-one"])
    with pytest.raises(AssertionError):
        llm.generate(["p1", "p2"], max_tokens=5, temperature=0.0, stop=None)
