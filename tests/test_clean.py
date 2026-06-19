from rag.indexer.clean import compute_boilerplate, clean_text, drop_junk

def test_compute_boilerplate_flags_repeated_lines():
    docs = ["Сервисы\nНомер счёта 20 знаков", "Сервисы\nКредитная карта", "Сервисы\nИпотека"]
    bp = compute_boilerplate(docs, min_doc_freq=3)
    assert "Сервисы" in bp

def test_clean_text_removes_boilerplate_and_collapses_ws():
    out = clean_text("Сервисы\n\n\nНомер   счёта\n\n", boilerplate={"Сервисы"})
    assert "Сервисы" not in out
    assert "Номер счёта" in out
    assert "\n\n\n" not in out

def test_drop_junk_filters_short_docs():
    assert drop_junk("x", min_len=20) is None
    assert drop_junk("a" * 25, min_len=20) is not None
