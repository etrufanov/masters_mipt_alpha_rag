from rag.generator.abstain import ABSTAIN, is_abstain, normalize_output

def test_abstain_constant():
    assert ABSTAIN == "Нет ответа."

def test_is_abstain_variants():
    assert is_abstain("Нет ответа")
    assert is_abstain("  нет ответа.  ")
    assert not is_abstain("Номер счёта состоит из 20 знаков.")

def test_normalize_empty_to_abstain():
    assert normalize_output("") == ABSTAIN
    assert normalize_output("   ") == ABSTAIN

def test_normalize_abstain_phrase_to_canonical():
    assert normalize_output("нет ответа") == ABSTAIN

def test_normalize_keeps_real_answer_stripped():
    assert normalize_output("  Ответ: счёт 20 знаков.  ") == "Ответ: счёт 20 знаков."
