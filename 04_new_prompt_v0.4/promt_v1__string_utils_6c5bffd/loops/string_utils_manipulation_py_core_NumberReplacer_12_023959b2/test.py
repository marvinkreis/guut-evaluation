from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function to ensure it correctly converts the number 1000 into 'M'.
    The mutant has altered the internal mappings for encoding Roman numerals, causing the function to fail for this input.
    The baseline returns 'M' for 1000, while the mutant raises a KeyError due to the incorrect mapping.
    """
    output = roman_encode(1000)
    assert output == 'M'