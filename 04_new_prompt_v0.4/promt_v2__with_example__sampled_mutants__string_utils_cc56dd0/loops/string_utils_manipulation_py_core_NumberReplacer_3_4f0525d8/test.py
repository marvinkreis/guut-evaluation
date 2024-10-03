from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 4. The baseline will return 'IV' correctly,
    while the mutant will raise a KeyError due to an incorrect mapping for Roman numeral encoding.
    """
    output = roman_encode(4)
    assert output == 'IV', f"Expected 'IV', got {output}"