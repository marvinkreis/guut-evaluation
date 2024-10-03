from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the value 500. The baseline should return 'D', 
    while the mutant will produce a KeyError due to the altered mappings.
    """
    output = roman_encode(500)
    assert output == 'D', f"Expected 'D', got {output}"