from rag.generator.length import length_unit_count

def test_word_count():
    assert length_unit_count("один два три", "words") == 3

def test_char_count():
    assert length_unit_count("абв", "chars") == 3
