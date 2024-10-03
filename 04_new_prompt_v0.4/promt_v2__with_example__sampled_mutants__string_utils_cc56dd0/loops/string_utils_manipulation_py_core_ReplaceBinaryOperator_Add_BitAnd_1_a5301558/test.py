from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 6. The expected output for the baseline is 'VI'.
    The mutant will raise a TypeError because of the use of the '&' operator, which is invalid for strings.
    """
    output = roman_encode(6)
    assert output == 'VI', f"Expected 'VI', got {output}"