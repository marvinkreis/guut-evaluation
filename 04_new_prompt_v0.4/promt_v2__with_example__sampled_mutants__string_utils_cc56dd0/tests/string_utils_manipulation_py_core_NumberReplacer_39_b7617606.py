from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with inputs that trigger the mutant's failure point. 
    The baseline will produce 'IX' for input 9 and 'XLIX' for input 49, while the mutant will raise a KeyError. 
    This shows the mutant's failure to encode the digit 9 correctly.
    """
    output_9 = roman_encode(9)
    assert output_9 == "IX", f"Expected 'IX', got {output_9}"

    output_49 = roman_encode(49)
    assert output_49 == "XLIX", f"Expected 'XLIX', got {output_49}"