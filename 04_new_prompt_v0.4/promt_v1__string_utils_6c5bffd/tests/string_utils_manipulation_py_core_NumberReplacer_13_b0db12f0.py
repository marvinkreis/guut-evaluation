from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with values that should map to 'M'.
    This test checks the correct conversion of numbers to Roman numerals. The input 1000 should yield 'M', and 2000 should yield 'MM'.
    The mutant is expected to fail this test due to an invalid mapping that leads to a KeyError.
    """
    output_1000 = roman_encode(1000)
    assert output_1000 == 'M', f"Expected 'M', got {output_1000}"
    
    output_2000 = roman_encode(2000)
    assert output_2000 == 'MM', f"Expected 'MM', got {output_2000}"