from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant may return incorrect values for the Roman numeral encoding, specifically for 6."""
    output = roman_encode(6)
    assert output == "VI", "Encoding for 6 must return 'VI'. Any different output indicates the mutant."