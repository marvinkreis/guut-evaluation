from string_utils.manipulation import roman_decode

def test_roman_decode_mutant_killing():
    """
    Test the roman_decode function with a valid Roman numeral 'IX', which should raise
    a TypeError in the mutant due to the invalid condition change in the logic. 
    The baseline will successfully return 9.
    """
    output = roman_decode('IX')
    assert output == 9, f"Expected 9, got {output}"