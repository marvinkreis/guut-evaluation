from shunting_yard import shunting_yard

def test__shunting_yard():
    """The mutant will not include operators in the output for the RPN conversion."""
    output = shunting_yard([10, '-', 5, '-', 2])
    # The output should contain operators between the operands
    assert output == [10, 5, '-', 2, '-'], "shunting_yard must include operators in the RPN"