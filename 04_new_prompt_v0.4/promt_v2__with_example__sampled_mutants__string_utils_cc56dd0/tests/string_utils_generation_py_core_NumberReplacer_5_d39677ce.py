from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function with the default start value.
    The mutant will raise ValueError because the start value is 0,
    which is not a valid input, while the baseline will return Roman numerals I to V.
    """
    output = list(roman_range(5))
    assert output == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {output}"