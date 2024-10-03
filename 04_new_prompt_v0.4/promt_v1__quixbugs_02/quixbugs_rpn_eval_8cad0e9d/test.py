from rpn_eval import rpn_eval

def test__rpn_eval_kill_mutant():
    """
    This test checks the evaluation of an RPN expression that involves
    subtraction and division: [5.0, 3.0, '-', 2.0, '/'].
    We expect it to return 1.0 in the baseline and -1.0 in the mutant,
    thus killing the mutant.
    """
    output = rpn_eval([5.0, 3.0, '-', 2.0, '/'])
    assert output == 1.0, f"Expected 1.0 but got {output}"