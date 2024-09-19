from rpn_eval import rpn_eval

def test__rpn_eval():
    """The mutant changes the order of arguments in the operation function, which disrupts correct RPN evaluation."""
    # Using a known RPN expression where the correct output is 4.0.
    output = rpn_eval([3.0, 5.0, '+', 2.0, '/'])
    assert output == 4.0, "The result must be 4.0 for the RPN expression [3.0, 5.0, '+', 2.0, '/']"