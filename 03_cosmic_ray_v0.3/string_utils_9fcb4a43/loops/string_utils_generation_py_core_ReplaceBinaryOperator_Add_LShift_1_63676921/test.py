from string_utils.generation import roman_range

def test__roman_range():
    """The mutant fails for roman_range(5, 3, 1) where it should return valid Roman numerals."""
    output = list(roman_range(5, 3, 1))
    assert output == ['III', 'IV', 'V'], "roman_range should yield ['III', 'IV', 'V'] for valid inputs"