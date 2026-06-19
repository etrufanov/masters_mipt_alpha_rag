import pytest
from rag.eval.metric import length_gate, combined_score

def test_length_gate_breakpoints():
    assert length_gate(1.0) == 1.0
    assert length_gate(1.5) == 1.0
    assert length_gate(3.0) == 0.0
    assert length_gate(4.0) == 0.0
    assert length_gate(2.25) == pytest.approx(0.5, abs=1e-9)   # -2/3*2.25+2

def test_combined_score_applies_gate_to_recall():
    # ours 3 words vs ref 1 word => ratio 3 => gate 0
    assert combined_score(recall=0.9, ours="a b c", ref="a", unit="words") == 0.0
    # equal length => gate 1
    assert combined_score(recall=0.8, ours="a b", ref="c d", unit="words") == pytest.approx(0.8)
