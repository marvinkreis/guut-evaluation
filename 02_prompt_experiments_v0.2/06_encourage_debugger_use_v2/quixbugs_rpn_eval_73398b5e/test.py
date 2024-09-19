from rpn_eval import rpn_eval

def test__rpn_eval():
    """The argument order in the op function affects results in non-commutative operations."""
    output = rpn_eval([5.0, 3.0, '-', 2.0, '/'])
    assert output == 1.0, "Expected output for (5.0, 3.0, '-', 2.0, '/') should be 1.0"