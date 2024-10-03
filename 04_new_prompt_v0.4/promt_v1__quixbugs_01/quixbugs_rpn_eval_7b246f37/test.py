from rpn_eval import rpn_eval

def test__rpn_eval():
    """
    Test the evaluation of Reverse Polish Notation with a set of tokens that produces different results
    based on operand order. The input represents the calculation (4 / 2 - 1), which will lead to different 
    results if the operand order is swapped, as shown in the mutant.
    """
    output = rpn_eval([4.0, 2.0, '/', 1.0, '-'])
    assert output == 1.0  # The expected output for the baseline