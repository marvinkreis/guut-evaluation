from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function for the case where we attempt to generate
    Roman numerals from 7 to 1 with a negative step. The baseline should
    yield the Roman numerals ['VII', 'VI', 'V', 'IV', 'III', 'II', 'I'],
    while the mutant will raise an OverflowError due to invalid configuration.
    """
    output = list(roman_range(1, 7, -1))
    assert output == ['VII', 'VI', 'V', 'IV', 'III', 'II', 'I']