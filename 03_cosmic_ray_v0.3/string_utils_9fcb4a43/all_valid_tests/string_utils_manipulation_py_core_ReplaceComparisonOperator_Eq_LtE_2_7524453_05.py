from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Testing roman encoding for boundary numbers influenced by mutant changes."""

    # Test for value 9, expected 'IX'
    output_9 = roman_encode(9)
    assert output_9 == "IX", "roman_encode should return 'IX' for the input 9"