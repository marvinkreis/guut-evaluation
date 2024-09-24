from string_utils.generation import roman_range

def test__roman_range():
    """The mutant allows an invalid backward range configuration and returns unexpected results."""
    output = list(roman_range(3, 5, -1))
    # Validate that the expected output is ['V', 'IV', 'III'] for a valid sequence
    assert output == ['V', 'IV', 'III'], "Mutant may return incorrect sequence of Roman numerals."