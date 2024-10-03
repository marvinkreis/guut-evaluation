from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 1234. The baseline will return the correct Roman numeral `MCCXXXIV`, 
    while the mutant returns an incorrect value `MCCXXIII`.
    """
    output = roman_encode(1234)
    assert output == 'MCCXXXIV', f"Expected 'MCCXXXIV', got {output}"