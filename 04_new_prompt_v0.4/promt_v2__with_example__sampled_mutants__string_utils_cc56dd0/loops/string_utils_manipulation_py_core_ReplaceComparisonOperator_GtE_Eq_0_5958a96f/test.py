from string_utils.manipulation import roman_decode

def test_roman_decode_mutant_killing():
    """
    Test the roman_decode function with two inputs:
    - 'XIII', which should return 13,
    - 'XL', which should return 40.
    
    The mutant's conditional change is expected to cause it to return incorrect values for both cases.
    """
    # Testing XIII
    output1 = roman_decode("XIII")
    assert output1 == 13, f"Expected 13 but got {output1}"

    # Testing XL
    output2 = roman_decode("XL")
    assert output2 == 40, f"Expected 40 but got {output2}"