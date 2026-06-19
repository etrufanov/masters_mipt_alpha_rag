from rag.schema import Chunk, Retrieved

def test_chunk_fields():
    c = Chunk(chunk_id="5:0", web_id=5, url="u", title="t", text="body")
    assert c.chunk_id == "5:0" and c.web_id == 5

def test_retrieved_is_chunk_with_score():
    r = Retrieved(chunk_id="5:0", web_id=5, url="u", title="t", text="body", score=0.9)
    assert isinstance(r, Chunk)
    assert r.score == 0.9
