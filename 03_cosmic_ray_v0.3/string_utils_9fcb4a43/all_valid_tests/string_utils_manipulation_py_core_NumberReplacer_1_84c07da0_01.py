from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's change in the mappings causes it to raise an error or produce incorrect output when encoding 1."""
    output = roman_encode(1)
    assert output == 'I', "roman_encode must return 'I' for input 1"