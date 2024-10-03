from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    This test checks the encoding of the number 10 to its Roman numeral representation. 
    The input 10 should correctly encode to 'X'. The mutant incorrectly changes the mapping 
    for tens, resulting in an incorrect encoding when 10 is passed as an argument.
    """
    output = roman_encode(10)
    assert output == 'X'