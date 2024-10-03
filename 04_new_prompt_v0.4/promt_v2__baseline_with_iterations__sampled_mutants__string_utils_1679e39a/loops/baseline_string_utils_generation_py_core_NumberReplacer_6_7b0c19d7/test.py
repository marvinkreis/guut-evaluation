from string_utils.generation import roman_range

def test__roman_range():
    """
    Test that the roman_range function behaves correctly when given a specific stop value.
    The input should generate roman numerals from 1 to 5, which will return 'I', 'II', 'III', 'IV', 'V'.
    Changing the step from 1 to 2 in the mutant will cause it to skip the even numbered roman numerals,
    leading to the failed expectation.
    """
    output = list(roman_range(5))
    assert output == ['I', 'II', 'III', 'IV', 'V']