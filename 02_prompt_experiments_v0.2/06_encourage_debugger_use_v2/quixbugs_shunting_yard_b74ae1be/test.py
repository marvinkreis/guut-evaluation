from shunting_yard import shunting_yard

def test__shunting_yard():
    """Removing opstack.append(token) from shunting_yard prevents proper RPN output by omitting operators."""
    output = shunting_yard([10, '-', 5, '-', 2])
    # Verify that the output contains the expected operators, thus failing for the mutant
    assert output == [10, 5, '-', 2, '-'], "Incorrect RPN output; the mutant fails to include operators."