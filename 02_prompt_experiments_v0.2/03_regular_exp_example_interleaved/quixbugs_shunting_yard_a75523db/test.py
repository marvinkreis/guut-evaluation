from shunting_yard import shunting_yard

def test__shunting_yard():
    """Removing the operator appending will cause the output to lack operators in RPN format."""
    output = shunting_yard([3, '+', 4])
    assert output == [3, 4, '+'], "shunting_yard must include operators in the output"