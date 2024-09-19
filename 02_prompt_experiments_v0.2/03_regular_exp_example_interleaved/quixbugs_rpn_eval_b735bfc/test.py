from rpn_eval import rpn_eval

def test__rpn_eval():
    """Changing the order of operands in rpn_eval would cause incorrect results for operations."""
    output = rpn_eval([5.0, 3.0, '-', 2.0, '/'])
    # The correct output should be 1.0
    assert output == 1.0, "rpn_eval must return the correct result when evaluated with [5.0, 3.0, '-', 2.0, '/']"