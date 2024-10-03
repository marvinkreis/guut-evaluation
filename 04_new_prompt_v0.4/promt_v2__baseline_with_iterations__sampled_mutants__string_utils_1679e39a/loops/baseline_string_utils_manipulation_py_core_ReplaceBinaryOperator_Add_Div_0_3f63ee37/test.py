from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the encoding of the number 4 returns the correct Roman numeral 'IV'.
    The mutant changes the logic to perform a division instead of a concatenation for the value 4, 
    which will lead to an error, making this test fail.
    """
    output = roman_encode(4)
    assert output == 'IV'