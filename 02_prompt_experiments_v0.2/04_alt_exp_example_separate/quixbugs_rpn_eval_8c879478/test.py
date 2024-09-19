from rpn_eval import rpn_eval

def test__rpn_eval():
    """Changing the operand order in the operation would lead to incorrect results in RPN evaluation."""
    output = rpn_eval([3.0, 5.0, '-', 2.0, '/'])
    assert output == -1.0, "RPN evaluation must correctly handle order of operations especially for subtraction and division."