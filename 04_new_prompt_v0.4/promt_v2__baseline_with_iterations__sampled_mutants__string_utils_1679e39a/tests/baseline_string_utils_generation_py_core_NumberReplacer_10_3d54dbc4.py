from string_utils.generation import roman_range

def test__roman_range():
    """
    This test checks if the function correctly raises a ValueError when the argument is out of range.
    The input is 4000, which exceeds the valid maximum of 3999. In the mutant, the check for the upper bound
    is incorrectly set to 4000, thus the code won't raise an exception as it should.
    """
    try:
        list(roman_range(4000))
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999'
    else:
        assert False, "ValueError not raised"