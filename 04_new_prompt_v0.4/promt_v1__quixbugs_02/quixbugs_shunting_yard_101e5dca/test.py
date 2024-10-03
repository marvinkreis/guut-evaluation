from shunting_yard import shunting_yard

def test__shunting_yard():
    """
    Test the conversion of an infix expression to RPN. The input [5, '+', 3, '*', 2]
    should yield [5, 3, 2, '*', '+'] in RPN. The mutant should fail to produce this output due to
    missing operator handling.
    """
    output = shunting_yard([5, '+', 3, '*', 2])
    assert output == [5, 3, 2, '*', '+']