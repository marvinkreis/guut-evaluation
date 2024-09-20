from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant changes the mapping for 'M' leading to a KeyError when encoding 1000."""
    output = roman_encode(1000)
    assert output == 'M', f"Expected 'M', but got {output}"