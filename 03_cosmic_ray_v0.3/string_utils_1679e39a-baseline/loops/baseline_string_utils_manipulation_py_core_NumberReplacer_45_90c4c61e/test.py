from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the encoded roman numeral correctly raises an exception when given a non-positive integer.
    The input value of 0 is invalid for roman numbers, and this will trigger the ValueError in the original code,
    while the mutant will not trigger it, allowing the input to pass through incorrectly.
    """
    try:
        roman_encode(0)
        assert False, "Expected ValueError for input 0"
    except ValueError:
        pass