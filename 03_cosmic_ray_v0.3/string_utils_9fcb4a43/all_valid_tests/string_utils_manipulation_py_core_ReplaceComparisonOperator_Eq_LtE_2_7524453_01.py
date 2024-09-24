from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant changes encoding for the value 4, which should return 'IV'."""
    output = roman_encode(4)
    assert output == "IV", "roman_encode should return 'IV' for the input 4"