from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the encoding of the number 20 generates the correct Roman numeral. The input '20' should return 'XX'.
    If the mutant incorrectly changes the mapping for tens, it will return an incorrect value (like 'XX').
    """
    output = roman_encode(20)
    assert output == 'XX'