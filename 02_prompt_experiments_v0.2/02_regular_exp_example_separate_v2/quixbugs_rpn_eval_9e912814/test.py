from rpn_eval import rpn_eval

def test__rpn_eval():
    """By changing the order of operands in rpn_eval, the mutant would return the incorrect result for non-commutative operations."""
    output = rpn_eval([5.0, 3.0, '-', 4.0, '/'])
    assert output == 0.5, "The evaluation did not return the expected result"