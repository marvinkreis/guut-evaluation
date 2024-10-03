from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input number 5.
    The baseline will correctly return 'V', while the mutant will raise an IndexError
    due to an invalid access to the mappings array.
    """
    output = roman_encode(5)
    assert output == 'V', f"Expected 'V', got {output}"