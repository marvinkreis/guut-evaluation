from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function using the value 1000. The baseline should return 'M',
    while the mutant will trigger a KeyError due to incorrect mapping for thousands.
    """
    output = roman_encode(1000)
    assert output == 'M', f"Expected 'M', got {output}"