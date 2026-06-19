from rag.generator.prompt import build_prompt
from rag.schema import Retrieved

def _ctx():
    return [Retrieved(chunk_id="1:0", web_id=1, url="u", title="Счета", text="Счёт 20 знаков.", score=0.9)]

def test_prompt_contains_query_context_and_abstain_instruction():
    p = build_prompt("Номер счета", _ctx())
    assert "Номер счета" in p
    assert "Счёт 20 знаков." in p
    assert "Нет ответа." in p          # abstain instruction present
    assert "Счета" in p                # title included
