from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the number 400.
    The baseline returns 'CD', while the mutant should raise a KeyError due to the invalid mapping for hundreds.
    """
    output = roman_encode(400)
    assert output == 'CD', f"Expected 'CD', got {output}"