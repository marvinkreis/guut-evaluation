from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input value of 10.
    The baseline will return 'X', while the mutant will raise a KeyError due to an incorrect mapping for tens in the mutant code.
    """
    output = roman_encode(10)
    assert output == 'X', f"Expected 'X', got {output}"