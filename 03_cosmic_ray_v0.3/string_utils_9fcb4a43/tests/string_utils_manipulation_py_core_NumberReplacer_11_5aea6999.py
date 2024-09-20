from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's change in the Roman numeral mapping leads to a KeyError for input 500."""
    output = roman_encode(500)
    assert output == "D", "Expected output for 500 should be 'D'."