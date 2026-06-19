from rag.indexer.chunk import chunk_text

WORDS = lambda s: len(s.split())

def test_short_text_is_single_chunk():
    out = chunk_text("короткий текст про счёт", target=400, overlap=64, len_fn=WORDS)
    assert out == ["короткий текст про счёт"]

def test_chunks_respect_target_with_tolerance():
    text = "\n\n".join("слово " * 50 for _ in range(20))  # ~1000 words
    out = chunk_text(text, target=100, overlap=20, len_fn=WORDS)
    assert len(out) > 1
    assert all(WORDS(c) <= 150 for c in out)   # target + tolerance

def test_overlap_present_between_consecutive_chunks():
    text = "\n".join(f"пункт{i} это предложение номер {i}." for i in range(60))
    out = chunk_text(text, target=40, overlap=10, len_fn=WORDS)
    a, b = set(out[0].split()), set(out[1].split())
    assert a & b   # some shared tokens => overlap
