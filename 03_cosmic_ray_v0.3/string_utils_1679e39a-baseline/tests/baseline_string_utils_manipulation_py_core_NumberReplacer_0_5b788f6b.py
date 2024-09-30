from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the encoding of the number 1 returns 'I'. The mutant changes the mapping for units, which 
    affects how the value is encoded. If the input is 1, the original code will correctly return 'I' 
    but the mutant will not, leading to a failure in this test.
    """
    output = roman_encode(1)
    assert output == 'I'