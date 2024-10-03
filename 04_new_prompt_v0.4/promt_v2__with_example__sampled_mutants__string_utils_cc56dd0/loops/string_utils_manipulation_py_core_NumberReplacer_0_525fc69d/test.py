from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function using the value 1.
    The mutant will raise a KeyError because it cannot find the required mapping,
    while the baseline will return 'I'.
    """
    output_1 = roman_encode(1)
    assert output_1 == 'I', f"Expected 'I', got {output_1}"