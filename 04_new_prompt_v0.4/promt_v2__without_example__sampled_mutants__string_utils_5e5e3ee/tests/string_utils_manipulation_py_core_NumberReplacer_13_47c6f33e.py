from string_utils.manipulation import roman_encode

def test__roman_encode_1000():
    """
    Test encoding the number 1000 which should return 'M' in proper Roman numeral encoding.
    The mutant modifies the encoding logic and should result in a KeyError.
    """
    output = roman_encode(1000)
    assert output == 'M'