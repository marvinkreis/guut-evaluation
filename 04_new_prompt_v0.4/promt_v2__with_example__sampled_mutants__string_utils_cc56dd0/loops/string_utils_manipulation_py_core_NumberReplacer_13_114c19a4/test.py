from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input number 1000.
    The baseline will return 'M', while the mutant will produce a KeyError
    due to the incorrect mapping in the encoded values.
    """
    output = roman_encode(1000)
    assert output == 'M', f"Expected 'M', got {output}"