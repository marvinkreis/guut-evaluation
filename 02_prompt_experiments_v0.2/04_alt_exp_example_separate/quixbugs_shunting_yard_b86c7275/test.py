from shunting_yard import shunting_yard

def test__shunting_yard():
    """Removing 'opstack.append(token)' will result in missing operators in the output."""
    output = shunting_yard([10, '-', 5, '-', 2])
    assert output == [10, 5, '-', 2, '-'], "shunting_yard must correctly convert infix to RPN"