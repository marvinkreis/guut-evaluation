from string_utils.generation import roman_range

def test__roman_range_negative_step():
    """
    Test the roman_range function with a negative step to verify correct handling of reverse iterations.
    The input represents a range where we expect to generate roman numerals from 5 down to 1.
    The expected output for `roman_range(1, 5, -1)` should be: ['V', 'IV', 'III', 'II', 'I'].
    The mutant, however, raises an OverflowError due to incorrect boundary checking.
    """
    output = list(roman_range(stop=1, start=5, step=-1))
    assert output == ['V', 'IV', 'III', 'II', 'I']