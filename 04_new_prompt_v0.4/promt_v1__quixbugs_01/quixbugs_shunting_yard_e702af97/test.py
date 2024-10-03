from shunting_yard import shunting_yard

def test__shunting_yard_operator_precedence():
    """
    Test the shunting yard algorithm with multiple operators to ensure it respects operator precedence and outputs correct RPN. 
    The input [3, '+', 5, '*', 2] should convert to RPN as [3, 5, 2, '*', '+'], demonstrating that multiplication comes before addition. 
    The baseline should return the expected output while the mutant will fail to do so.
    """
    tokens = [3, '+', 5, '*', 2]
    output = shunting_yard(tokens)
    print(f"output = {output}")

    assert output == [3, 5, 2, '*', '+']