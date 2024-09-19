from shunting_yard import shunting_yard

def test__shunting_yard():
    """The mutant omitting opstack.append(token) would result in missing operators in the output."""
    output = shunting_yard([3, '+', 4, '*', 2])
    print(f"output = {output}")
    assert len(output) > 3, "The output must include both operands and operators."