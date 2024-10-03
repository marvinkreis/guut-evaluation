from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 100.
    The mutant will raise a KeyError due to the incorrect mapping in hundreds,
    while the baseline will correctly return 'C'.
    """
    output = roman_encode(100)
    assert output == 'C', f"Expected 'C', got {output}"